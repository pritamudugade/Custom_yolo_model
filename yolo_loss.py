

import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv3Loss(nn.Module):
    def __init__(self, num_classes, anchors, img_size=416):
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors)  # The anchor boxes, provided as a list of tuples (width, height)
        self.num_anchors = len(anchors)
        self.img_size = img_size

    def forward(self, predictions, targets):
        # Unpack the predictions
        pred_boxes, pred_conf, pred_cls = predictions

        # Get the batch size
        batch_size = pred_boxes.size(0)

        # Calculate the grid size based on the size of the prediction feature map
        grid_size = pred_boxes.size(3)

        # Compute the number of bounding box attributes (x, y, w, h, objectness score)
        num_attributes = 5

        # Reshape the predictions to have the shape (batch_size, num_anchors, num_attributes, grid_size, grid_size)
        pred_boxes = pred_boxes.view(batch_size, self.num_anchors, num_attributes, grid_size, grid_size)

        # Get the target values for each attribute
        tgt_boxes = targets[:, :, 2:]
        tgt_conf = targets[:, :, 0]
        tgt_cls = targets[:, :, 1]

        # Calculate the anchor boxes in the original image space
        scaled_anchors = torch.tensor([(a_w / self.img_size, a_h / self.img_size) for a_w, a_h in self.anchors],
                                      dtype=pred_boxes.dtype, device=pred_boxes.device)

        # Calculate the target x, y, w, h values in the grid cell space
        tgt_xy = targets[:, :, 2:4] * grid_size
        tgt_wh = targets[:, :, 4:] * grid_size

        # Calculate the predicted x, y, w, h values in the grid cell space
        pred_xy = pred_boxes[:, :, :2].sigmoid()
        pred_wh = pred_boxes[:, :, 2:4].exp() * scaled_anchors

        # Calculate the predicted bounding box coordinates
        pred_box = torch.cat((pred_xy, pred_wh), dim=-1)

        # Calculate the confidence loss using binary cross-entropy
        obj_mask = tgt_conf.unsqueeze(-1).expand_as(pred_conf)
        no_obj_mask = 1 - obj_mask
        conf_loss = F.binary_cross_entropy_with_logits(pred_conf, tgt_conf, reduction='none')
        conf_loss = obj_mask * conf_loss + no_obj_mask * conf_loss

        # Calculate the localization loss using mean squared error
        loc_loss = F.mse_loss(pred_box * obj_mask, tgt_boxes * obj_mask, reduction='none').sum(dim=-1)

        # Calculate the class loss using cross-entropy
        cls_loss = F.cross_entropy(pred_cls, tgt_cls, reduction='none')

        # Compute the total loss as a sum of all components
        loss = conf_loss + loc_loss + cls_loss

        # Take the mean over all samples in the batch
        loss = loss.mean()

        return loss





