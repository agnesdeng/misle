
#'midae cost function
midae_optimizer<-function(x,na_idx,networkOutput,output_split,output_struc){
  tf <- tensorflow::tf
  tf$compat$v1$disable_eager_execution()
  na_split=tf$split(na_idx,output_split,axis= as.integer(1))
  true_split=tf$split(x,output_split,axis= as.integer(1))
  x_reconstr_mean<-networkOutput$x_reconstr_mean
  x_reconstr_logvar<-networkOutput$x_reconstr_logvar
  pred_split=tf$split(x_reconstr_mean,output_split,axis=as.integer(1))
  cost_list=list()
  for (n in 1:length(output_struc)){
    na_adj=tf$cast(tf$math$count_nonzero(na_split[[n]]),tf$float32)/tf$cast(tf$size(na_split[[n]]),tf$float32)
    if (output_struc[n] == 'numeric'){

      cost_list[[n]]=tf$sqrt(tf$compat$v1$losses$mean_squared_error(tf$boolean_mask(true_split[[n]],na_split[[n]]),tf$boolean_mask(pred_split[[n]],na_split[[n]])))*na_adj

    }
    else if (output_struc[n] == 'binary'){
      cost_list[[n]]=tf$compat$v1$losses$sigmoid_cross_entropy(tf$boolean_mask(true_split[[n]],na_split[[n]]),
                                                     tf$boolean_mask(pred_split[[n]],na_split[[n]]))*na_adj
    }
    else {
      cost_list[[n]]=tf$compat$v1$losses$softmax_cross_entropy(tf$reshape(tf$boolean_mask(true_split[[n]],na_split[[n]]),shape=shape(-1,output_struc[n])),
                                                     tf$reshape(tf$boolean_mask(pred_split[[n]],na_split[[n]]),shape=shape(-1,output_struc[n])))*na_adj
    }
  }




  log_var=0.1*x_reconstr_logvar
  loss1=tf$reduce_mean(tf$multiply(tf$exp(-log_var),tf$reduce_sum(cost_list)))
  loss2=tf$reduce_mean(log_var)
  joint_loss<-0.5*(loss1+loss2)



  return(joint_loss)
}


# output function
output_function<-function(reconstr_output,output_split,output_struc){
  tf <- tensorflow::tf
  out_split=tf$split(reconstr_output,output_split,axis=as.integer(1))
  output_list=list()
  for (n in 1:length(output_struc)){
    if(output_struc[n]=="numeric"){
      output_list[[n]]=out_split[n]
    }else if(output_struc[n]=="binary"){
      output_list[[n]]=tf$nn$sigmoid(out_split[n])
    }else{
      output_list[[n]]=tf$nn$softmax(out_split[n])
    }
  }
  output_list

}


#mivae cost function
mivae_optimizer<-function(x, na_idx,networkOutput,output_split,output_struc,kld=TRUE,loss_weight=1,kld_weight=1,sigma_weight=1){
  tf <- tensorflow::tf
  na_split=tf$split(na_idx,output_split,axis= as.integer(1))
  true_split=tf$split(x,output_split,axis= as.integer(1))
  x_reconstr_mean<-networkOutput$x_reconstr_mean
  x_reconstr_logvar<-networkOutput$x_reconstr_logvar
  z_log_sigma_sq<-networkOutput$z_log_sigma_sq
  z_mean<-networkOutput$z_mean
  z<-networkOutput$z
  pred_split=tf$split(x_reconstr_mean,output_split,axis=as.integer(1))
  cost_list=list()
  for (n in 1:length(output_struc)){
    na_adj=tf$cast(tf$math$count_nonzero(na_split[[n]]),tf$float32)/tf$cast(tf$size(na_split[[n]]),tf$float32)

    if (output_struc[n] == 'numeric'){

      cost_list[[n]]=tf$compat$v1$losses$mean_squared_error(tf$boolean_mask(true_split[[n]],na_split[[n]]),tf$boolean_mask(pred_split[[n]],na_split[[n]]))*(1/(na_adj+1e-5))

    }
    else if (output_struc[n] == 'binary'){
      cost_list[[n]]=tf$compat$v1$losses$sigmoid_cross_entropy(tf$boolean_mask(true_split[[n]],na_split[[n]]),
                                                     tf$boolean_mask(pred_split[[n]],na_split[[n]]))*na_adj
    }
    else {
      cost_list[[n]]=tf$compat$v1$losses$softmax_cross_entropy(tf$reshape(tf$boolean_mask(true_split[[n]],na_split[[n]]),shape=shape(-1,output_struc[n])),
                                                     tf$reshape(tf$boolean_mask(pred_split[[n]],na_split[[n]]),shape=shape(-1,output_struc[n])))*na_adj
    }
  }



  log_var=sigma_weight*x_reconstr_logvar
  loss1=tf$reduce_mean(tf$multiply(tf$exp(-log_var),tf$reduce_sum(cost_list)))

  loss2=tf$reduce_mean(log_var)

  joint_loss<-0.5*(loss1+loss2)





  #KLD
   loss_latent<--0.5*tf$reduce_sum(1+z_log_sigma_sq-tf$square(z_mean)-tf$exp(z_log_sigma_sq), axis=shape(1))


  if(kld){

    cost = tf$reduce_mean(loss_weight*joint_loss+kld_weight*loss_latent)
  }else{
    cost = tf$reduce_mean(joint_loss)
  }


  return(cost)
}
