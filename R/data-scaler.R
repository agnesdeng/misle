
#'scale a vector using minmax
#'@export
minmax <- function(x){
  (x-min(x))/(max(x)-min(x))
}


#'scale a dataset and return a scaled dataframe, the colmin and colmax of each column
#'@export
minmax_scaler=function(data){
  pre=dplyr::mutate_all(data,~ifelse(is.na(.),median(.,na.rm=TRUE),.))
  colmin=apply(pre,2,min)
  colmax=apply(pre,2,max)
  minmax.obj=NULL
  minmax.obj$minmax.df<-apply(pre,2,minmax)
  minmax.obj$colmin<-colmin
  minmax.obj$colmax<-colmax
  return(minmax.obj)
}





#'mimax data function
#'@export
minmax_data <- function(data){
  colmin=apply(data,2,min)
  colmax=apply(data,2,max)
  minmax_data=NULL
  minmax_data$minmax.df<-apply(data,2,minmax)
  minmax_data$colmin<-colmin
  minmax_data$colmax<-colmax
  return(minmax_data)
}




#'This function back-transform data to an output as data matrix
#'@export
inv.minmax_data<-function(data,colmin,colmax){
  output<-sapply(seq(1, ncol(data), by=1),
                 FUN=function(x, data, colmin, colmax)
                 {
                   (data[, x]*(colmax[x]-colmin[x])+colmin[x])
                 }, data, colmin, colmax)
  return(output)
}
