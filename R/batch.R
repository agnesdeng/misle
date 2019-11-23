#' make batch datasets, each with size batch_size
#' @param data A dataset
#' @param batch_size Batch size
#' @return This function returns \code{num_batch} datasets with size equals \code{batch_size} each

batch_iter<-function(data,batch_size){
  n=nrow(data)
  if(batch_size>n){
    stop("Batch size should be smaller than the size of data")
  }
  index=sample(1:n)
  num_batch=n %/% batch_size
  batchset=list()
  start_idx=seq(1,n-batch_size+1,batch_size)
  for(i in 1:num_batch){
    batchset[[i]]=index[start_idx[i]:(start_idx[i]+batch_size-1)]
  }
  return(batchset)
}




