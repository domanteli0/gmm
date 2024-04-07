#' @export
to_device <- function(x, device) {
  x$to_device(device)
}

#' @export
to_device.default <- function(x, device) {
  x$to_device(device)
}
