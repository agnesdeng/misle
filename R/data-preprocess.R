

#'This function is used to detect the type(numeric,binary,multiclass) of each feature
#' @param  data A data frame
#' @export
variable_class<-function(data){
  binary<-NULL
  multiclass<-NULL
  feature<-colnames(data)
  for(i in 1:ncol(data)){
    if(length(levels(data[,i]))==2){
      binary=c(binary,feature[i])
    }else if(length(levels(data[,i]))>2){
      multiclass=c(multiclass,feature[i])
    }
  }
  return(list("binary"=binary,"multiclass"=multiclass))
}




#'This function is used to convert dataframe into onehot dataframe
#'It also returns \code{column_list}, each list component contains classes of one non-numeric feature
#'@param data a data frame
#'@param autolabel logical; if TRUE the class of variables would be automatically found and users are not required to provide the names of \code{binary} and \code{multiclass}
#'@param binary A vector contains names of binary features
#'@param multiclass A vector contains names of multiclass features
#'@export
onehot<-function(data,autolabel=TRUE,binary=NULL,multiclass=NULL){
  if(autolabel==TRUE){
    binary<-variable_class(data)$binary
    multiclass<-variable_class(data)$multiclass
  }
  fac=c(binary,multiclass)
  cat=names(data) %in% fac
  fac.df=data[fac]
  num.df=data[!cat]
  onehot.df=num.df
  columns_list=list()
  for(column in colnames(fac.df)){
    na_temp =is.na(fac.df[column])
    temp = sjmisc::to_dummy(fac.df[column],suffix="label")
    onehot.df=cbind(onehot.df,temp)
    columns_list[[column]]=colnames(temp)
  }
  return(list("onehot.df"=onehot.df, "columns_list"=columns_list))
}


#' This function is used to sequence the columns of the dataset, so as to be in the order Numeric>Binary>Multiclass
#' It is used to minimise memory overhead
#' @param data A data frame
#' @param feature name of the feature (character)
#' @return This function returns a list with \code{sorted.df} and \code{chunk} (the number of feature in the selected group)
#' @export
sort_feature=function(data,feature){
  group=colnames(data) %in% feature
  data_1 <- data[group]
  data_0 <- data[!group]
  chunk <- ncol(data_1)
  sorted.df <- cbind(data_0,data_1)
  return(list("sorted.df"= sorted.df, "chunk" =chunk))
}


#' This function is use to identify the structure of input
#' @param data A data frame (not one hot yet)
#' @param columns_list A list, each list component contains classes of one non-numeric feature
#' @param binary A vector contains names of binary features
#' @param multiclass A vector contains names of multiclass features
#' @param unstorted A logical value indicating whether the data is sorted or not
#' @param verbose A logical value indicating whether information should be printed out
#' @return This function return a list with \code{size_index} and \code{output_struc}
#' @export
output_structure=function(data,columns_list=NULL,binary=NULL,multiclass=NULL,unsorted=TRUE,verbose=FALSE){
  onehot.df=onehot(data)$onehot.df
  #if the dataframe has no missing value, raise an error "data contains no missing values"
  if(is.null(columns_list)){
    columns_list=onehot(data)$columns_list
  }

  #name of feature
  feature_name <- colnames(onehot.df)
  num_exists <- FALSE
  fac_exists <- FALSE

  #number of feature
  feature_size <- ncol(onehot.df)

  #Cost functions for difference variables
  #preallocate empty space? or use append
  size_index=NULL


  #binary features
  if(is.null(binary)){
    binary<-variable_class(data)$binary
  }
  if(is.null(binary)==FALSE){
    if (unsorted) {
      for (feature in binary){
        S <-sort_feature(onehot.df,columns_list[[feature]])
        onehot.df <- S$sorted.df
        chunk <-S$chunk
        size_index <- append(size_index,chunk)
      }
    }else{
      size_index <- append(size_index,binary)
    }
    binary_exists <-TRUE

  }else{
    binary_exists <-FALSE
  }




  #multiclass features
  if(is.null(multiclass)){
    multiclass<-variable_class(data)$multiclass
  }
  if(is.null(multiclass)==FALSE){
    if (unsorted){
      for (feature in multiclass){
        S <-sort_feature(onehot.df,columns_list[[feature]])
        onehot.df <- S$sorted.df
        chunk <-S$chunk
        size_index <- append(size_index,chunk)
      }

    }else{
      size_index <- append(size_index,multiclass)
    }
    multiclass_exists <-TRUE
    }else{
    multiclass_exists <-FALSE
  }

  #numeric features
  if (sum(size_index) < feature_size){
    chunk <- feature_size-sum(size_index)
    size_index <- append(size_index, chunk, after=0)
    num_exists <-TRUE
    if (sum(size_index) != feature_size)
      stop("Sorting columns has failed")
  }

  #verbose
  if (verbose){
    cat("Size index:", size_index,"\n")
  }


  output_struc <-NULL

   if (num_exists){
        output_struc <- rep('numeric',size_index[1])
   }else if (binary_exists){
        output_struc <- 'binary'
   }else{
        output_struc <-size_index[1]
      }

  if(length(size_index)>=2){
    for (i in 2:length(size_index)){
      if (size_index[i]==2) {
        output_struc <- c(output_struc,'binary')
      }else{
        output_struc <- c(output_struc, size_index[i])
      }
    }

  }


  output_split=NULL
  for (i in 1:length(output_struc)){
    if( output_struc[i] == "numeric"){
      output_split <- append(output_split,as.integer(1))
    }else if(output_struc[i] == "binary"){
      output_split <- append(output_split,as.integer(2))
    }
    else{
      output_split <-append(output_split,as.integer(output_struc[i]))
    }
  }

  output<-list()
  output$size_index<-size_index
  output$output_struc<-output_struc
  output$output_split<-output_split
  output$onehot.df<-onehot.df
  output$columns_list<-columns_list
  output
  }


#'Sort the dataset by increasing number of missing values
#'@param data A data frame (with missing values NA's)
#'@export

sortNA<-function(data){
  na.loc=is.na(data)
  sorted.idx<-order(colSums(na.loc))
  sorted.df<-data[sorted.idx]
  return(list("sorted.df"=sorted.df,"sorted.idx"=sorted.idx))
}


#'Return the type of each variable in the dataset
#'@param data A data frame
#'@export

feature_type<-function(data){
  type<-rep("numeric",ncol(data))
  binary<-variable_class(data)$binary
  multiclass<-variable_class(data)$multiclass
  if(is.null(binary)==FALSE){
    binary.index<-which(colnames(data) %in% binary)
    type[binary.index]<-"binary"
  }

  if(is.null(multiclass)==FALSE){
    multiclass.index<-which(colnames(data) %in% multiclass)
    type[multiclass.index]<-"multiclass"
  }
  return(type)
}




