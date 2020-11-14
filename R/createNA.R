#' Create missing value for a dataset
#' @param data a dataframe
#' @param p the proportion of missing values in the dataframe
#' @param seed random seed, a singlue value
#' @export
createNA <- function(data,p=0.1,seed=2019){
  Nrow=nrow(data)
  Ncol=ncol(data)
  if(length(p)==1){
    total<-Nrow*Ncol
    NAloc <- rep(FALSE, total)
    set.seed(seed)
    NAloc[sample(total, floor(total * p))] <- TRUE
    data[matrix(NAloc, nrow = Nrow, ncol = Ncol)] <- NA
  }else{
    for(i in 1:length(p)){
      data[,i][sample(Nrow,round(p[i]*Nrow))]<-NA
    }

  }
  return(data)

}

