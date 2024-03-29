#' R6 class for Midae imputer
#' @docType  class
#' @description Set up a Midae imputer object with specified hyperparameters and then obtain multiple imputed datasets
#' @format  NULL
#' @import tensorflow
#' @export






Midae<- R6::R6Class("Midae",
                     public = list(
                       #input_size
                       n_input =NULL,
                       #latent_size
                       n_h = NULL,
                       encoder_structure = NULL,
                       decoder_structure = NULL,
                       #learn_rate
                       learn_rate = NULL,
                       #ITERATION
                       iteration = NULL,
                       #BATCH
                       batch_size = NULL,
                       saver = NULL,
                       vae_graph =NULL,
                       ###
                       x = NULL,
                       na_idx  = NULL,
                       network_weights = NULL,
                       networkOutput = NULL,
                       output_op = NULL,
                       cost = NULL,
                       optimizer = NULL,
                       input_drop=NULL,
                       hidden_drop=NULL,
                       output_struc=NULL,
                       output_split=NULL,
                       data=NULL,
                       onehot.df=NULL,
                       bootstrap.df=NULL,
                       columns_list=NULL,
                       act=NULL,
                       weight_decay=NULL,
                       rho=NULL,
                       epsilon=NULL,
                       decay=NULL,
                       momentum=NULL,
                       initialize = function(data,
                                             n_h = 2L,
                                             encoder_structure = c(128L,64L,32L),
                                             decoder_structure = "reversed",
                                             learn_rate = 1e-4,
                                             iteration= 400,
                                             batch_size = 500,
                                             input_drop=0.2,
                                             hidden_drop=0,
                                             #act=tf$nn$softplus,
                                             act=tf$nn$elu,
                                             weight_decay="default",
                                             epsilon=1e-7,
                                             rho=0.95,
                                             decay=0.9,
                                             momentum=0.9,
                                             optimizer="Adam") {
                         tf <- tensorflow::tf
                         Out<-output_structure(data)
                         self$onehot.df<-Out$onehot.df
                         self$columns_list<-Out$columns_list
                         self$data<-data
                         n_input<-ncol(self$onehot.df)
                         self$output_struc<-Out$output_struc
                         self$output_split<-Out$output_split
                         self$n_input<- as.integer(n_input)
                         self$n_h <- as.integer(n_h)
                         self$encoder_structure <- as.integer(encoder_structure)

                         if(decoder_structure == "reversed"){
                           self$decoder_structure <-as.integer(rev(encoder_structure))
                         }else{
                           self$decoder_structure <- as.integer(decoder_structure)
                         }

                         if(weight_decay =="default"){
                           lambda<-1/nrow(self$data)
                         }else{
                           lambda<-weight_decay
                         }
                         self$weight_decay<-lambda
                         self$epsilon<-epsilon
                         self$rho<-rho
                         self$decay<-decay
                         self$momentum<-momentum

                         self$learn_rate  <- learn_rate
                         self$iteration <- iteration
                         self$batch_size  <- batch_size
                         self$input_drop<-input_drop
                         self$hidden_drop<-hidden_drop
                         self$act<-act

                         #tf$reset_default_graph()
                         self$vae_graph <- tf$Graph()

                         with(self$vae_graph$as_default(),{
                           tf <- tensorflow::tf
                           tf$compat$v1$disable_eager_execution()
                           tf$compat$v1$disable_v2_behavior()


                           self$x = tf$compat$v1$placeholder(tf$float32, shape=shape(NULL, self$n_input), name='x')
                           self$na_idx=tf$compat$v1$placeholder(tf$bool,shape=shape(NULL, self$n_input),name='na_idx')
                           self$network_weights<-midae_init(encoder_structure=self$encoder_structure,decoder_structure=self$decoder_structure,n_input=self$n_input,n_h=self$n_h)
                           self$networkOutput<-midae_output(self$act,self$x, self$network_weights,encoder_structure=self$encoder_structure,decoder_structure=self$decoder_structure,
                                                            self$input_drop,self$hidden_drop)
                           self$output_op<-output_function(self$networkOutput$x_reconstr_mean,self$output_split,self$output_struc)
                           self$cost=midae_optimizer(self$x,self$na_idx,self$networkOutput,self$output_split,self$output_struc)

                           if(optimizer=="Adam"){
                             self$optimizer = tf$compat$v1$train$AdamOptimizer(learning_rate=self$learn_rate,epsilon=self$epsilon)$minimize(self$cost)
                           }else if(optimizer=="AdamW"){
                             self$optimizer = tf$contrib$opt$AdamWOptimizer(weight_decay=self$weight_decay,learning_rate=self$learn_rate,epsilon=self$epsilon)$minimize(self$cost)
                           }else if(optimizer=="Adadelta"){
                             self$optimizer = tf$compat$v1$train$AdadeltaOptimizer(learning_rate=self$learn_rate,rho=self$rho,epsilon=self$epsilon)$minimize(self$cost)
                           }else if(optimizer=="RMSProp"){
                             self$optimizer = tf$compat$v1$train$RMSPropOptimizer(learning_rate=self$learn_rate,decay=self$decay,momentum=self$momentum,epsilon=self$epsilon)$minimize(self$cost)
                           }


                           self$saver=tf$compat$v1$train$Saver()
                         })
                       },

                       train=function(verbose=TRUE,print_freq=1){
                         tf <- tensorflow::tf
                         tf$compat$v1$disable_v2_behavior()
                         data=self$onehot.df


                         #scale data and get colmin and colmax
                         scaled.obj=minmax_scaler(data)
                         scaled.mat<-scaled.obj$minmax.df
                         #mark the location of nonmising values (not NA)
                         notna_loc<-!is.na(data)
                         idx <- which(is.na(data))
                         scaled.mat[idx]<-0

                         with(tf$compat$v1$Session(graph=self$vae_graph) %as% sess,{
                           sess$run(tf$compat$v1$global_variables_initializer())
                           num_batch=(nrow(scaled.mat) %/% (self$batch_size))
                           batchset=batch_iter(scaled.mat,self$batch_size)

                           for(k in 1:self$iteration){
                             current_loss<-0
                             for(i in 1:num_batch){
                               b<-batchset[[i]]
                               x<-self$x
                               na_idx<-self$na_idx
                               batch_train<-sess$run(list(self$cost,self$optimizer),feed_dict = dict(x=scaled.mat[b,],na_idx=notna_loc[b,]))
                               current_loss<-current_loss+batch_train[[1]]

                             }
                             current_loss<-current_loss/num_batch

                             if(verbose==TRUE){
                               if (k %% print_freq == 0){
                                 cat("Iteration - ", k, "Current Loss - ", current_loss,"\n")
                                 #self$saver$save(sess,"Temp/Midae.ckpt")
                               }
                             }



                           }

                           self$saver$save(sess,"Temp/Midae.ckpt")
                         })
                       },


                       impute=function(m=5,onehot=FALSE,all.numeric=FALSE,add.noise=FALSE,SD=1,pmm=FALSE){
                         tf <- tensorflow::tf
                         data=self$onehot.df
                         #scale data and get colmin and colmax
                         scaled.obj=minmax_scaler(data)
                         scaled.mat<-scaled.obj$minmax.df
                         scaled.df <-as.data.frame(scaled.mat)
                         colmin<-scaled.obj$colmin
                         colmax<-scaled.obj$colmax
                         #mark the location of nonmising values (not NA)
                         notna_loc<-!is.na(data)
                         idx <- which(is.na(data))
                         scaled.mat[idx]<-0
                         with(tf$compat$v1$Session(graph=self$vae_graph) %as% sess,{
                           sess$run(tf$compat$v1$global_variables_initializer())
                           self$saver$restore(sess,"Temp/Midae.ckpt")
                           x <- self$x
                           imputed.data<-list()
                           onehot.data<-list()
                           if(pmm){
                              #future work..hasn't finished yet
                             for(i in 1:m){
                               output.list<-sess$run(self$output_op, feed_dict = dict(x=scaled.mat))
                               output.mat<-matrix(unlist(output.list),ncol=self$n_input)
                               temp<-inv.minmax_data(output.mat,colmin,colmax)
                               onehot.data[[i]]=temp
                               onehot.data[[i]]=as.matrix(onehot.data[[i]],ncol=self$n_input)
                               colnames(onehot.data[[i]])=colnames(data)
                             }

                           }else{
                             for(i in 1:m){
                               output.list<-sess$run(self$output_op, feed_dict = dict(x=scaled.mat))
                               output.mat<-matrix(unlist(output.list),ncol=self$n_input)
                               temp<-inv.minmax_data(output.mat,colmin,colmax)
                               onehot.data[[i]]=as.matrix(data)
                               onehot.data[[i]][!notna_loc]<-temp[!notna_loc]
                               onehot.data[[i]]=as.matrix(onehot.data[[i]],ncol=self$n_input)
                               colnames(onehot.data[[i]])=colnames(data)
                             }

                           }



                         })

                         if(add.noise){
                           onehot.data=lapply(onehot.data,function(x) x+rnorm(nrow(self$data*self$n_input),0,sd=SD))
                         }




                         if(all.numeric){

                           return(lapply(onehot.data,as.data.frame))


                         }else if(onehot){
                           return(onehot.data)
                         }
                         else{
                           for(i in 1:m){
                             imputed.data[[i]]<-inverse_onehot(onehot.data[[i]],self$data,columns_list=self$columns_list)
                           }
                           return(imputed.data)
                         }

                       },
                       impute.new=function(newdata=data,m=5,onehot=FALSE,all.numeric=FALSE,pmm=FALSE){
                         tf <- tensorflow::tf

                         Out<-output_structure(newdata)
                         onehot.df<-Out$onehot.df
                         data=onehot.df
                         #scale data and get colmin and colmax
                         scaled.obj=minmax_scaler(data)
                         scaled.mat<-scaled.obj$minmax.df
                         scaled.df <-as.data.frame(scaled.mat)
                         colmin<-scaled.obj$colmin
                         colmax<-scaled.obj$colmax
                         #mark the location of nonmising values (not NA)
                         notna_loc<-!is.na(data)
                         idx <- which(is.na(data))
                         scaled.mat[idx]<-0
                         with(tf$compat$v1$Session(graph=self$vae_graph) %as% sess,{
                           sess$run(tf$compat$v1$global_variables_initializer())
                           self$saver$restore(sess,"Temp/Midae.ckpt")
                           x <- self$x
                           imputed.data<-list()
                           onehot.data<-list()
                           if(pmm){
                             #future work..hasn't finished yet
                             for(i in 1:m){
                               output.list<-sess$run(self$output_op, feed_dict = dict(x=scaled.mat))
                               output.mat<-matrix(unlist(output.list),ncol=self$n_input)
                               temp<-inv.minmax_data(output.mat,colmin,colmax)
                               onehot.data[[i]]=temp
                               onehot.data[[i]]=as.matrix(onehot.data[[i]],ncol=self$n_input)
                               colnames(onehot.data[[i]])=colnames(data)
                             }

                           }else{
                             for(i in 1:m){
                               output.list<-sess$run(self$output_op, feed_dict = dict(x=scaled.mat))
                               output.mat<-matrix(unlist(output.list),ncol=self$n_input)
                               temp<-inv.minmax_data(output.mat,colmin,colmax)
                               onehot.data[[i]]=as.matrix(data)
                               onehot.data[[i]][!notna_loc]<-temp[!notna_loc]
                               onehot.data[[i]]=as.matrix(onehot.data[[i]],ncol=self$n_input)
                               colnames(onehot.data[[i]])=colnames(data)
                             }

                           }



                         })





                         if(all.numeric){

                           return(lapply(onehot.data,as.data.frame))


                         }else if(onehot){
                           return(onehot.data)
                         }
                         else{
                           for(i in 1:m){
                             imputed.data[[i]]<-inverse_onehot(onehot.data[[i]],newdata,columns_list=self$columns_list)
                           }
                           return(imputed.data)
                         }

                       }


                     )
)





