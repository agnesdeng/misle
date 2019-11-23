#'utility functions


is_windows <- function() {
  identical(.Platform$OS.type, "windows")
}
