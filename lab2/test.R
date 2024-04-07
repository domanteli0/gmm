library(torch)
library(torchvision)
library(torchdatasets)

library(dplyr)
library(pins)
library(ggplot2)

device <- torch_device("mps")

train_transforms <- function(img) {
  img %>%
    # first convert image to tensor
    transform_to_tensor() %>%
    # then move to the GPU (if available)
    (function(x) x$to(device = device)) %>%
    # data augmentation
    transform_random_resized_crop(size = c(224, 224), scale = c(0.75, 0.75)) %>%
    # data augmentation
    transform_color_jitter() %>%
    # data augmentation
    transform_random_horizontal_flip() %>%
    # normalize according to what is expected by resnet
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}

valid_transforms <- function(img) {
  img %>%
    transform_to_tensor() %>%
    (function(x) x$to(device = device)) %>%
    transform_resize(256) %>%
    transform_center_crop(224) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}

test_transforms <- valid_transforms

train_ds <- image_folder_dataset(
  file.path("OIDv6", "train"),
  transform = train_transforms
)

test_ds <- image_folder_dataset(
  file.path("OIDv6", "test"),
  transform = train_transforms
)

batch_size <- 24

class_names <- train_ds$classes
length(class_names)

train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)

batch <- train_dl$.iter()$.next()

classes <- batch[[2]]
classes

images <- as_array(batch[[1]]) %>% aperm(perm = c(1, 3, 4, 2))
mean <- c(0.485, 0.456, 0.406)
std <- c(0.229, 0.224, 0.225)
images <- std * images + mean
images <- images * 255
images[images > 255] <- 255
images[images < 0] <- 0

par(mfcol = c(4,6), mar = rep(1, 4))

images %>%
  purrr::array_tree(1) %>%
  purrr::set_names(class_names[as_array(classes)]) %>%
  purrr::map(as.raster, max = 255) %>%
  purrr::iwalk(~{plot(.x); title(.y)})
  