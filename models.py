from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

from parse_config import *
from utils import build_targets
from collections import defaultdict

##import matplotlib.pyplot as plt
##import matplotlib.patches as patches


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module("route_%d" % i, EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None):
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals)

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        self.losses["recall"] /= 3
        self.losses["precision"] /= 3
        return sum(output) if is_training else torch.cat(output, 1)

    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    """

    def save_weights(self, path, cutoff=-1):

        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


from utils import compute_boxes_and_sizes, get_upsample_output, get_box_and_dot_maps, get_boxed_img


class LSCCNN(nn.Module):
    def __init__(self, name='scale_4', checkpoint_path=None, output_downscale=2,
                 PRED_DOWNSCALE_FACTORS=(8, 4, 2, 1), GAMMA=(1, 1, 2, 4), NUM_BOXES_PER_SCALE=3):

        super(LSCCNN, self).__init__()
        self.name = name
        if torch.cuda.is_available():
            self.rgb_means = torch.cuda.FloatTensor([104.008, 116.669, 122.675])
        else:
            self.rgb_means = torch.FloatTensor([104.008, 116.669, 122.675])
        self.rgb_means = torch.autograd.Variable(self.rgb_means, requires_grad=False).unsqueeze(0).unsqueeze(
            2).unsqueeze(3)

        self.BOXES, self.BOX_SIZE_BINS = compute_boxes_and_sizes(PRED_DOWNSCALE_FACTORS, GAMMA, NUM_BOXES_PER_SCALE)
        self.output_downscale = output_downscale

        in_channels = 3
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.convA_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convA_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convA_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convA_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convA_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.convB_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.convB_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convB_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convB_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convB_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.convC_1 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.convC_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convC_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convC_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convC_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.convD_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convD_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convD_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convD_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convD_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.conv_before_transpose_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.transpose_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_1_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.transpose_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_2_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.transpose_3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.conv_after_transpose_3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.transpose_4_1_a = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.transpose_4_1_b = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_4_1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)

        self.transpose_4_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.conv_after_transpose_4_2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)

        self.transpose_4_3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_4_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.conv_middle_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv_middle_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_middle_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_mid_4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.conv_lowest_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_lowest_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_lowest_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_lowest_4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.conv_scale1_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_scale1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_scale1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path))

    def forward(self, x):
        mean_sub_input = x
        mean_sub_input -= self.rgb_means

        #################### Stage 1 ##########################

        main_out_block1 = self.relu(self.conv1_2(self.relu(self.conv1_1(mean_sub_input))))
        main_out_pool1 = self.pool1(main_out_block1)

        main_out_block2 = self.relu(self.conv2_2(self.relu(self.conv2_1(main_out_pool1))))
        main_out_pool2 = self.pool2(main_out_block2)

        main_out_block3 = self.relu(self.conv3_3(self.relu(self.conv3_2(self.relu(self.conv3_1(main_out_pool2))))))
        main_out_pool3 = self.pool3(main_out_block3)

        main_out_block4 = self.relu(self.conv4_3(self.relu(self.conv4_2(self.relu(self.conv4_1(main_out_pool3))))))
        main_out_pool4 = self.pool3(main_out_block4)

        main_out_block5 = self.relu(self.conv_before_transpose_1(
            self.relu(self.conv5_3(self.relu(self.conv5_2(self.relu(self.conv5_1(main_out_pool4))))))))

        main_out_rest = self.convA_5(self.relu(
            self.convA_4(self.relu(self.convA_3(self.relu(self.convA_2(self.relu(self.convA_1(main_out_block5)))))))))
        if self.name == "scale_1":
            return main_out_rest
        ################## Stage 2 ############################

        sub1_out_conv1 = self.relu(self.conv_mid_4(self.relu(
            self.conv_middle_3(self.relu(self.conv_middle_2(self.relu(self.conv_middle_1(main_out_pool3))))))))
        sub1_transpose = self.relu(self.transpose_1(main_out_block5))
        sub1_after_transpose_1 = self.relu(self.conv_after_transpose_1_1(sub1_transpose))

        sub1_concat = torch.cat((sub1_out_conv1, sub1_after_transpose_1), dim=1)

        sub1_out_rest = self.convB_5(self.relu(
            self.convB_4(self.relu(self.convB_3(self.relu(self.convB_2(self.relu(self.convB_1(sub1_concat)))))))))
        if self.name == "scale_2":
            return main_out_rest, sub1_out_rest
        ################# Stage 3 ############################

        sub2_out_conv1 = self.relu(self.conv_lowest_4(self.relu(
            self.conv_lowest_3(self.relu(self.conv_lowest_2(self.relu(self.conv_lowest_1(main_out_pool2))))))))
        sub2_transpose = self.relu(self.transpose_2(sub1_out_conv1))
        sub2_after_transpose_1 = self.relu(self.conv_after_transpose_2_1(sub2_transpose))

        sub3_transpose = self.relu(self.transpose_3(main_out_block5))
        sub3_after_transpose_1 = self.relu(self.conv_after_transpose_3_1(sub3_transpose))

        sub2_concat = torch.cat((sub2_out_conv1, sub2_after_transpose_1, sub3_after_transpose_1), dim=1)

        sub2_out_rest = self.convC_5(self.relu(
            self.convC_4(self.relu(self.convC_3(self.relu(self.convC_2(self.relu(self.convC_1(sub2_concat)))))))))

        if self.name == "scale_3":
            return main_out_rest, sub1_out_rest, sub2_out_rest

        ################# Stage 4 ############################
        sub4_out_conv1 = self.relu(
            self.conv_scale1_3(self.relu(self.conv_scale1_2(self.relu(self.conv_scale1_1(main_out_pool1))))))

        # TDF 1
        tdf_4_1_a = self.relu(self.transpose_4_1_a(main_out_block5))
        tdf_4_1_b = self.relu(self.transpose_4_1_b(tdf_4_1_a))
        after_tdf_4_1 = self.relu(self.conv_after_transpose_4_1(tdf_4_1_b))

        # TDF 2
        tdf_4_2 = self.relu(self.transpose_4_2(sub1_out_conv1))
        after_tdf_4_2 = self.relu(self.conv_after_transpose_4_2(tdf_4_2))

        # TDF 3
        tdf_4_3 = self.relu(self.transpose_4_3(sub2_out_conv1))
        after_tdf_4_3 = self.relu(self.conv_after_transpose_4_3(tdf_4_3))

        sub4_concat = torch.cat((sub4_out_conv1, after_tdf_4_1, after_tdf_4_2, after_tdf_4_3), dim=1)
        sub4_out_rest = self.convD_5(self.relu(
            self.convD_4(self.relu(self.convD_3(self.relu(self.convD_2(self.relu(self.convD_1(sub4_concat)))))))))

        if self.name == "scale_4":
            return main_out_rest, sub1_out_rest, sub2_out_rest, sub4_out_rest

    def predict_single_image(self, image, nms_thresh=0.25, thickness=2, multi_colours=True):
        if image.shape[0] % 16 or image.shape[1] % 16:
            image = cv2.resize(image, (image.shape[1]//16*16, image.shape[0]//16*16))
        img_tensor = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(img_tensor.cuda())
        out = get_upsample_output(out, self.output_downscale)
        pred_dot_map, pred_box_map = get_box_and_dot_maps(out, nms_thresh, self.BOXES)
        img_out = get_boxed_img(image, pred_box_map, pred_box_map, pred_dot_map, self.output_downscale,
                                self.BOXES, self.BOX_SIZE_BINS, thickness=thickness, multi_colours=multi_colours)
        return pred_dot_map, pred_box_map, img_out

