#' Install the TensorFlow backend (modified code from Keras)
#'
#' TensorFlow will be installed in to an "r-tensorflow" virtual or conda environment.
#' Note: "virtualenv" is not avilable on Windows
#'
#' @inheritParams tensorflow::install_tensorflow
#'
#' @param method Installation method ("virtualenv" or "conda")
#'
#' @param conda_python_version the python version installed in the created conda
#'   environment. Python 3.6 is installed by default.
#'
#' @param tensorflow TensorFlow version to install. Specify "default" to install
#'   the CPU version of the latest release. Specify "gpu" to install the GPU
#'   version of the latest release.
#'
#' @param extra_packages Additional PyPI packages to install along with
#'   Keras and TensorFlow.
#'
#' @param ... Other arguments passed to [tensorflow::install_tensorflow()].
#'
#' @section GPU Installation:
#'
#' TensorFlow can be configured to run on either CPUs or GPUs. The CPU
#' version is much easier to install and configure so is the best starting place
#' especially when you are first learning how to use misle.
#'
#' - *TensorFlow with CPU support only*. If your system does not have a NVIDIA® GPU,
#' you must install this version. Note that this version of TensorFlow is typically
#' much easier to install, so even if you have an NVIDIA GPU, we recommend installing
#' this version first.
#'
#' - *TensorFlow with GPU support*. TensorFlow programs typically run significantly
#' faster on a GPU than on a CPU. Therefore, if your system has a NVIDIA® GPU meeting
#' all prerequisites and you need to run performance-critical applications, you should
#' ultimately install this version.
#'
#' To install the GPU version:
#'
#' 1) Ensure that you have met all installation prerequisites including installation
#'    of the CUDA and cuDNN libraries as described in [TensorFlow GPU Prerequistes](https://tensorflow.rstudio.com/installation_gpu.html#prerequisites).
#'
#' 2) Pass `tensorflow = "gpu"` to `install_misle()`. For example:
#'
#'     ```
#'       install_misle(tensorflow = "gpu")
#'     ````
#'
#' @section Windows Installation:
#'
#' The only supported installation method on Windows is "conda". This means that you
#' should install Anaconda 3.x for Windows prior to installing Keras.
#'
#'
#' @section Additional Packages:
#'
#' If you wish to add additional PyPI packages to your Keras / TensorFlow environment you
#' can either specify the packages in the `extra_packages` argument of `install_misle()`,
#' or alternatively install them into an existing environment using the
#' [reticulate::py_install()] function.
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
                          #python_version="3.6",
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


  extra_packages <- unique(c(
    paste0("keras", version),
    extra_packages,
    "h5py",
    "pyyaml",
    "requests",
    "Pillow",
    "scipy"
  ))

  # perform the install
  install_tensorflow(method = method,
                     conda = conda,
                     version = tensorflow,
                     extra_packages = extra_packages,
                     #conda_python_version = python_version,
                     pip_ignore_installed = FALSE,
                     ...)
}
