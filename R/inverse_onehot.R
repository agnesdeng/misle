
#inverse onehot function
inverse_onehot=function(onehot.mat,data,columns_list=NULL){
  np<-tensorflow::import("numpy")
  #data: original data
  #onehot.mat:  imputed data in matrix format, (without any Missing values)

  #numeric data
  id<-which(sapply(data,is.numeric))
  data[,id]<-onehot.mat[,names(id)]

  ##calling onehot make it slow
  if(is.null(columns_list)){
    columns_list=onehot(data)$columns_list
  }
  feature.names<-names(columns_list)

  #binary and multiclass (categorical)
  for (i in 1:length(columns_list)){
    cat=colnames(onehot.mat) %in% columns_list[[i]]
    y=onehot.mat[,cat]
    ylevels<-gsub(".*_", "", colnames(y))
    result<-np$argmax(y,axis=as.integer(1))
    result<-result+1
    data[,feature.names[i]]<-factor(ylevels[result],labels=ylevels)
  }
  return(data)
}
