
import torch
from PIL import Image
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import NoiseTunnel
from captum.attr import Saliency
from captum.attr import visualization as viz
from captum.attr import Occlusion

# helper class

color = [(0, '#ffffff'), (0.25, '#000000'), (1, '#000000')]
name = 'custom blue'
N = 256

def linear_seg_color_map(name, color, N, gamma=1.0):
    """
    Render color map based on lookup tables
    :param name: name the color
    :param color: color code
    :param N: number of RGB quantization
    :param gamma: default is 1.0
    :return:
    """
    default_cmap = LinearSegmentedColormap.from_list(name, color, N, gamma)
    return default_cmap

def predict(model, transforms, image_path, use_cpu):
    """

    :param model:
    :param transforms: data transform
    :param image_path: inference input image path
    :param use_cpu:
    :return:
    """
    model.cpu()
    model = model.eval()
    img = Image.open(image_path)
    img = img.convert('RGB')
    transformed_img = transforms(img)
    image = transformed_img
    image = image.unsqueeze(0)
    if use_cpu:
        image = image.cpu()
    elif use_cpu == False:
        model.cuda()
        image = image.cuda
    output = model(image)

    output = torch.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    return image, transformed_img, prediction_score, pred_label_idx



def interpret_model(model, transforms, image_path, label_path, use_cpu=True, interpret_type=""):

    """
    :param model: our model
    :param transforms: Data transformation
    :param image_path: Image directory
    :param label_path: Json label directory
    :param use_gpu: set gpu to True
    :param interpret_type: mode for model interpretability: "integrated gradients"
    for Integrated Gradients, "gradient shap" for Gradient Shap and "occlusion" for Occlusion
    :return:
    """
    with open(label_path) as json_data:
        idx_to_labels = json.load(json_data)

    # Check if mode is Integrated Gradients
    if interpret_type == "integrated gradients":

        print('Performing Integrated Gradients Model Interpretation', interpret_type)
        image, transformed_img, prediction_score, pred_label_idx = predict(model, transforms, image_path, use_cpu)
        pred_label_idx.squeeze_()
        predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
        print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

        integrated_gradients = IntegratedGradients(model)
        attributions_ig = integrated_gradients.attribute(image, target=pred_label_idx, n_steps=20)

        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              ["original_image", "heat_map"],
                                              ["all", "absolute_value"],
                                              cmap=linear_seg_color_map(name, color, N),
                                              show_colorbar=True)

    # Check if mode is Integrated Gradients with Noise Tunnel
    elif interpret_type == "integrated gradient noise":
        print('Performing Integrated Gradients Noise Tunnel Model Interpretation', interpret_type)
        image, transformed_img, prediction_score, pred_label_idx = predict(model, transforms, image_path, use_cpu)
        pred_label_idx.squeeze_()
        predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
        print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

        integrated_gradients = IntegratedGradients(model)
        noise_tunnel = NoiseTunnel(integrated_gradients)

        attributions_ig_nt = noise_tunnel.attribute(image, n_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx)
        _ = viz.visualize_image_attr_multiple(
            np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            ["original_image", "heat_map"],
            ["all", "positive"],
            cmap=linear_seg_color_map(name, color, N),
            show_colorbar=True)



    # Check if mode is Gradient Shap
    elif interpret_type == "gradient shap":

        print('Performing Gradient Shap Model Interpretation', interpret_type)
        image, transformed_img, prediction_score, pred_label_idx = predict(model,transforms, image_path, use_cpu)
        pred_label_idx.squeeze_()
        predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
        print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
        gradient_shap = GradientShap(model)

        # Defining baseline distribution of images
        rand_img_dist = torch.cat([image * 0, image * 1])

        attributions_gs = gradient_shap.attribute(image,
                                                  n_samples=50,
                                                  stdevs=0.0001,
                                                  baselines=rand_img_dist,
                                                  target=pred_label_idx)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              ["original_image", "heat_map"],
                                              ["all", "absolute_value"],
                                              cmap=linear_seg_color_map(name, color, N),
                                              show_colorbar=True)

        # Check if mode is Occlusion
    elif interpret_type == "occlusion":

        print('Performing Occlusion Model Interpretation', interpret_type)
        image, transformed_img, prediction_score, pred_label_idx = predict(model, transforms, image_path, use_cpu)
        pred_label_idx.squeeze_()
        predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
        print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
        occlusion = Occlusion(model)

        # Defining baseline distribution of images
        rand_img_dist = torch.cat([image * 0, image * 1])

        attributions_gs = occlusion.attribute(image,
                                                  n_samples=50,
                                                  stdevs=0.0001,
                                                  baselines=rand_img_dist,
                                                  target=pred_label_idx)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              ["original_image", "heat_map"],
                                              ["all", "absolute_value"],
                                              cmap=linear_seg_color_map(name, color, N),
                                              show_colorbar=True)

        # Check if mode is Saliency
    elif interpret_type == "saliency":

        print('Performing Saliency Model Interpretation', interpret_type)
        image, transformed_img, prediction_score, pred_label_idx = predict(model, transforms, image_path, use_cpu)
        pred_label_idx.squeeze_()
        predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
        print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
        saliency = Saliency(model)
        attributions_gs = saliency.attribute(image,target=pred_label_idx)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                              ["original_image", "heat_map"],
                                              ["all", "absolute_value"],
                                              cmap=linear_seg_color_map(name, color, N),
                                              show_colorbar=True)




    






