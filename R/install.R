#' Install the TensorFlow backend (modified code from Keras)
#'
#' TensorFlow will be installed in to an "r-tensorflow" virtual or conda environment.
#' Note: "virtualenv" is not avilable on Windows#'
#' @inheritParams tensorflow::install_tensorflow
#'
#' @examples
#' \dontrun{
#'
#' # default installation
#' library(misle)
#' install_misle()
#'
#' # install using a conda environment (default is virtualenv)
#' install_misle(method = "conda")
#'
#' # install with GPU version of TensorFlow
#' # (NOTE: only do this if you have an NVIDIA GPU + CUDA!)
#' install_misle(tensorflow = "gpu")
#'
#' # install a specific version of TensorFlow
#' install_misle(tensorflow = "1.2.1")
#' install_misle(tensorflow = "1.2.1-gpu")
#'
#' }
#'
#' @importFrom reticulate py_available conda_binary
#'
#' @export
install_misle <- function(method = c("auto", "virtualenv", "conda"),
                          conda = "auto",
                          version = "default",
                          tensorflow = "default",
                          python_version="3.6",
                          extra_packages = c("tensorflow-hub"),
                          ...) {

  # verify method
  method <- match.arg(method)

  # resolve version
  if (identical(version, "default"))
    version <- ""
  else
    version <- paste0("==", version)

  # some special handling for windows
  if (is_windows()) {

    # conda is the only supported method on windows
    method <- "conda"

    # confirm we actually have conda
    have_conda <- !is.null(tryCatch(conda_binary(conda), error = function(e) NULL))
    if (!have_conda) {
      stop("Misle installation failed (no conda binary found)\n\n",
           "Install Anaconda for Python 3.x (https://www.anaconda.com/download/#windows)\n",
           "before installing Misle",
           call. = FALSE)
    }

    # avoid DLL in use errors
    if (py_available()) {
      stop("You should call install_misle() only in a fresh ",
           "R session that has not yet initialized misle and TensorFlow (this is ",
           "to avoid DLL in use errors during installation)")
    }
  }


  # perform the install
  install_tensorflow(method = method,
                     conda = conda,
                     version = tensorflow,
                     extra_packages = extra_packages,
                     conda_python_version = python_version,
                     #pip_ignore_installed = FALSE,
                     ...)
}
