# Army-Surveillance-Enhancer

This project introduces novel models designed from scratch to address the dual challenges of **detecting and enhancing pixelated images** with **exceptional speeds** targeted specifically for Military Surveillance. The proposed detection model, leveraging **MobileNet_v3_small combined with Canny edge detection**, demonstrates significant improvements over baseline methods on datasets like **Div2K** and **Flickr2K**. This model achieves higher precision, recall, F1 score, and accuracy while maintaining a lower false positive rate. 

## Members:
- Aviral Srivastava
- Kshitij Kumar
- Garv Bhaskar
- Shilpi Anand

## Faculty:
- Dr. Vergin Raja Sarobin

###  <ins>Detection Results: </ins>

#### MobileNet_v3_small + Canny Edge Detection

**Datasets used for testing:**

- Div2K (Full dataset - 900 images)
- Flickr2K (Test split - 284 images)

**Performance:**

The baseline model was not evaluated on the Div2K dataset due to its poor performance on the Flickr2K validation/test set.

## Training and Testing Dataset Details

#### Detector Training
The detector was trained on the train split of the Flickr2K dataset, which consists of 2,200 images.

#### Detector Testing
The detector was tested in two phases:
1. Test split of the Flickr2K dataset, consisting of 284 images.
2. The full dataset of Div2K (train + val) to ensure the images were entirely independent of the trained dataset.


## Inference: Detection

To run the detection app, use the following command:
```sh
python detector.py
```

## Contributing

Any kind of enhancement or contribution is welcomed.
