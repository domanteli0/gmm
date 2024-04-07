# # Lab2
# 
# ## Užduotis
# 
# Antroje užduotyje reikia realizuoti vaizdų klasifikavimo modelį.
# Atsiskaitinėjant pratybų dėstytojas atsiųs testinių vaizdų, su kuriais turėsite pademonstruoti, kaip jūsų realizuotas modelis veikia.
# Atsiskaitymo metu, turėsite gebėti papasakoti, kaip realizuotas, jūsų modelis.
# Programinės įrangos sprendimą galite naudoti savo nuožiūra.
# 
# - [ ] Klasės pasirenkamos savo nuožiūra, tačiau jų turi būti bent 3.
# - [ ] Duomenų rinkinys turi būti padalintas į mokymo ir testavimo aibes.
# - [ ] Su testavimo duomenų aibe reikia paskaičiuoti šias metrikas: klasifikavimo matrica (angl. *confusion matrix*), tikslumas, precizija, atkūrimas ir F1.
# 
# Duomenų klasėms parinktos iš [OpenImages V6](https://storage.googleapis.com/openimages/web/index.html) objektų aptikimo uždavinio duomenų rinkinio.
# 
# ## Įgyvendintų papildomų funkcijų papildomi balai $P_2$ pasirinktinai:
# 
# - [ ] Palyginimas palyginant aukšto lygio požymius (angl. _similiarity search_)
# - [ ] Sukuriant vartotojo sąsają ir modelio iškvietimą per REST API.
# 
# ## Užduoties įgyvendinimas
# 
# > [!NOTE]
# > Rekomenduoja projektą atidaryti naudojant [RStudio](https://posit.co/products/open-source/rstudio)
# 
# ### Duomenų atsiuntimas
# 
# ```{bash}
# brew install awscli
# ```
# 
# ```{bash}
# pipx install oidv6
# ```
# ```{bash}
# oidv6 downloader --classes Airplane Bus Boat Train --type_data test --no-labels --limit 500 --dest_dir OIDv6/test
# ```
# ```{bash}
# oidv6 downloader --classes Airplane Bus Boat Train --type_data train --no-labels --limit 200 --dest_dir OIDv6/
#   train
# ```
# ```{bash}
# oidv6 downloader --classes Airplane Bus Boat Train --type_data validation --no-labels --limit 200 --dest_dir OIDv6/
#   validation
# ```

### Dependencies


# install.packages("torch")
# install.packages("torchvision")
# install.packages("luz")
# install.packages("coro")
# install.packages('ggplot2')
# install.packages("imager")
# install.packages("todevice")


library(torchvision)
library(torch)
library(luz)
library(coro)
library(ggplot2)
library(imager)
install_torch()
library(todevice)

### Device

DEVICE = "mps"
device <- torch_device(DEVICE)

### Model


library(torch)

DIM = 32
CHANNELS = 3
NUM_CLASSES = 4

# Define the neural network architecture
net <- nn_module(
  
  initialize = function() {
    self$conv1 = nn_conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)
    self$batchnorm1 = nn_batch_norm2d(num_features = 32)
    self$conv2 = nn_conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)
    self$batchnorm2 = nn_batch_norm2d(num_features = 32)
    self$dropout1 = nn_dropout2d(p = 0.2)
    
    self$conv3 = nn_conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
    self$batchnorm3 = nn_batch_norm2d(num_features = 64)
    self$conv4 = nn_conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
    self$batchnorm4 = nn_batch_norm2d(num_features = 64)
    self$dropout2 = nn_dropout2d(p = 0.3)
    
    self$conv5 = nn_conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
    self$batchnorm5 = nn_batch_norm2d(num_features = 128)
    self$conv6 = nn_conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
    self$batchnorm6 = nn_batch_norm2d(num_features = 128)
    self$dropout3 = nn_dropout2d(p = 0.4)
    
    self$flatten = nn_flatten()
    self$fc1 = nn_linear(in_features = 128 * (DIM / 8) * (DIM / 8), out_features = 128)
    self$batchnorm_fc = nn_batch_norm1d(num_features = 128)
    self$dropout_fc = nn_dropout(p = 0.5)
    self$fc2 = nn_linear(in_features = 128, out_features = NUM_CLASSES)
  },
  
  forward = function(self, x) {
    x <- x %>%
      self$conv1() %>%
      torch_relu() %>%
      self$batchnorm1() %>%
      self$conv2() %>%
      torch_relu() %>%
      self$batchnorm2() %>%
      self$maxpool1() %>%
      self$dropout1()
    print(x$shape)
    
    x <- x %>%
      self$conv3() %>%
      torch$relu() %>%
      self$batchnorm3() %>%
      self$conv4() %>%
      torch$relu() %>%
      self$batchnorm4() %>%
      self$maxpool2() %>%
      self$dropout2() %>%
      print(x$shape)
    
    x <- x %>%
      torch$relu() %>%
      self$conv5() %>%
      self$batchnorm5() %>%
      self$conv6() %>%
      torch$relu() %>%
      self$batchnorm6() %>%
      self$maxpool3() %>%
      self$dropout3() %>%
      print(x$shape)
    
    x <- x %>%
      self$flatten() %>%
      self$fc1() %>%
      torch$relu() %>%
      self$batchnorm_fc() %>%
      self$dropout_fc() %>%
      self$fc2()
    print(x$shape)
    x
  }
)

model <- net()
model <- model$to(device = DEVICE)

### Displaying an image


library(imager)
display <- function(image) {
  # im <- load.image("myimage")
  # plot(im)
  
  # images <- train_dl$.iter()$.next()[[1]][1:32, 1, , ] 
  as_array(image) %>%
    purrr::array_tree(1) %>%
    purrr::map(as.raster) %>%
    purrr::iwalk(~{plot(.x)})
}


### Transformacijos

library(torchvision)

# Define data transformations using 'torchvision' package
train_trans <- function(img) {
  img %>%
    torchvision::transform_to_tensor() %>%
    torchvision::transform_random_horizontal_flip() %>%
    # torchvision::transform_random_rotation(20) %>%
    torchvision::transform_color_jitter(brightness = 0.4, contrast = 0.2, saturation = 0.2, hue = 0.1) %>%
    # torchvision::transform_random_grayscale(p = 0.1) %>%
    torchvision::transform_normalize(mean = c(0.5, 0.5, 0.5), std = c(0.5, 0.5, 0.5)) %>%
    torchvision::transform_random_resized_crop(size = c(32, 32), scale = c(0.75, 0.75))
}

validation_trans <- function(img) {
  img %>%
    torchvision::transform_to_tensor() %>%
    torchvision::transform_normalize(mean = c(0.5, 0.5, 0.5), std = c(0.5, 0.5, 0.5)) %>%
    torchvision::transform_resize(c(32, 32))
}

test_trans <- function(img) {
  img <- torchvision::transform_to_tensor(img)
  img <- img %>%
    torchvision::transform_grayscale(num_output_channels = 3) %>%
    torchvision::transform_resize(c(32, 32))
}

library(torchvision)
train_ds <- image_folder_dataset(
  file.path("OIDv6", "train"),
  transform = train_trans
)

test_ds <- image_folder_dataset(
  file.path("OIDv6", "test"),
  transform = test_trans
)

validation_ds <- image_folder_dataset(
  file.path("OIDv6", "validation"),
  transform = validation_trans
)

train_dl      <- dataloader(train_ds,      batch_size = 32, shuffle = TRUE,  num_workers = 2)
test_dl       <- dataloader(test_ds,       batch_size = 32, shuffle = FALSE, num_workers = 2)
validation_dl <- dataloader(validation_ds, batch_size = 32, shuffle = FALSE, num_workers = 2)

# Define loss function, optimizer, and device
criterion <- nn_cross_entropy_loss()
optimizer <- optim_adam(model$parameters, lr = 0.001)

# Training loop
num_epochs <- 30
for (epoch in 1:num_epochs) {
  # Set model to training mode
  model$train()
  
  # Initialize variables for tracking loss and accuracy
  train_loss_values <- c()
  train_corrects <- 0
  
  # Iterate over training dataset
  coro::loop(
    for (batch in test_dl) {
    #    inputs <- batch[[1]]$to(device)
    #    labels <- batch[[2]]$to(device)
    #    
    #    # Zero the gradients
    #    optimizer$zero_grad()
    #    
    #    # Forward pass
    #    outputs <- model(inputs)
    #    
    #    # Calculate loss
    #    loss <- criterion(outputs, labels)
    #    
    #    # Backward pass and optimization
    #    loss$backward()
    #    optimizer$step()
    #    
    #    # Track training loss
    #    train_loss_values <- c(train_loss_values, as.numeric(loss$cpu()$detach()))
    #    
    #    # Calculate number of correct predictions
    #    train_corrects <- train_corrects + sum(outputs$argmax(1) == labels)
  })
  
  # Calculate training accuracy
  train_acc <- train_corrects / length(train_data)
  
  # Print training loss and accuracy
  cat(sprintf("Epoch [%d/%d]: Train Loss: %.4f, Train Acc: %.4f\n", epoch, num_epochs, mean(train_loss_values), train_acc))
  
  # Validation loop
  model$eval()
  # Define variables for tracking validation loss and accuracy
  val_loss_values <- c()
  val_corrects <- 0
  
  # Iterate over validation dataset
  for (batch in validation_dl) {
    inputs <- batch[[1]]$to(device)
    labels <- batch[[2]]$to(device)
    
    # Forward pass
    outputs <- model(inputs)
    
    # Calculate loss
    loss <- criterion(outputs, labels)
    
    # Track validation loss
    val_loss_values <- c(val_loss_values, as.numeric(loss$cpu()$detach()))
    
    # Calculate number of correct predictions
    val_corrects <- val_corrects + sum(outputs$argmax(1) == labels)
  }
  
  # Calculate validation accuracy
  val_acc <- val_corrects / length(validation_data)
  
  # Print validation loss and accuracy
  cat(sprintf("Epoch [%d/%d]: Val Loss: %.4f, Val Acc: %.4f\n", epoch, num_epochs, mean(val_loss_values), val_acc))
}
