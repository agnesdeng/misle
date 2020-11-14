
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
    na_adj=tf$cast(tf$count_nonzero(na_split[[n]]),tf$float32)/tf$cast(tf$size(na_split[[n]]),tf$float32)

    if (output_struc[n] == 'numeric'){
      #cost_list[[n]]=tf$losses$mean_squared_error(tf$boolean_mask(true_split[[n]],na_split[[n]]),tf$boolean_mask(pred_split[[n]],na_split[[n]]))*na_adj

      #cost_list[[n]]=tf$multiply(tf$boolean_mask(true_split[[n]],na_split[[n]]),tf$log(1e-10+tf$boolean_mask(pred_split[[n]],na_split[[n]])))
                                 #+tf$multiply((1-tf$boolean_mask(true_split[[n]],na_split[[n]])),tf$log(1e-10+1-tf$boolean_mask(pred_split[[n]],na_split[[n]])))

      cost_list[[n]]=tf$losses$mean_squared_error(tf$boolean_mask(true_split[[n]],na_split[[n]]),tf$boolean_mask(pred_split[[n]],na_split[[n]]))*(1/(na_adj+1e-5))
      #cost_list[[n]]=tf$sqrt(tf$losses$mean_squared_error(tf$boolean_mask(true_split[[n]],na_split[[n]]),tf$boolean_mask(pred_split[[n]],na_split[[n]])))*na_adj
    }
    else if (output_struc[n] == 'binary'){
      cost_list[[n]]=tf$losses$sigmoid_cross_entropy(tf$boolean_mask(true_split[[n]],na_split[[n]]),
                                                     tf$boolean_mask(pred_split[[n]],na_split[[n]]))*na_adj
    }
    else {
      cost_list[[n]]=tf$losses$softmax_cross_entropy(tf$reshape(tf$boolean_mask(true_split[[n]],na_split[[n]]),shape=shape(-1,output_struc[n])),
                                                     tf$reshape(tf$boolean_mask(pred_split[[n]],na_split[[n]]),shape=shape(-1,output_struc[n])))*na_adj
    }
  }

  ####NEW
  ll_gaussian=function(sample,mu,log_var){
    sigma = tf$exp(0.5 * log_var)
    out=-0.5 * tf$log(2 * pi * sigma^2) - (1 / (2 * sigma^2))* (sample-mu)^2

    tf$reduce_sum(out,axis=as.integer(1))

  }

  #logp.x_z=ll_gaussian(y, mu, log_var)
  #logp.x_z=-tf$reduce_sum(cost_list)
  #log_var=z_log_sigma_sq
  #sigma = tf$exp(0.5 * log_var)
  #logp.x_z=ll_gaussian(x,mu=z_mean,log_var=z_log_sigma_sq)
  #####
  #logp.x_z=tf$reduce_sum(-0.5 * tf$log(2 * pi * sigma^2) - (1 / (2 * sigma^2))*loss_weight*tf$reduce_sum(cost_list),axis=as.integer(1))
  #logp.x_z=-tf$reduce_sum(cost_list)
 # logp.z=ll_gaussian(y_pred, 0, torch.log(torch.tensor(1.)))
  #logp.z=ll_gaussian(sample=z, 0, 0)
  #logq.z_x=ll_gaussian(sample=z, mu=z_mean, log_var=z_log_sigma_sq)
  #COST=-tf$reduce_mean(logp.x_z+ logp.z-logq.z_x)
  ####

   ##########CVAE
  #likelihood=tf$reduce_sum(x*tf$log(tf$clip_by_value(x_reconstr_mean,1e-10,x_reconstr_mean))+(1-x)*tf$log(tf$clip_by_value(1-x_reconstr_mean,1e-10,1)),reduction_indices=shape(1))
  #likelihood=tf$reduce_sum(x*tf$log(x_reconstr_mean)+(1-x)*tf$log(1-x_reconstr_mean),reduction_indices=shape(1))
 # likelihood=tf$reduce_sum(tf$multiply(x,tf$log(x_reconstr_mean+1e-12))+tf$multiply(1-x,tf$log(1-x_reconstr_mean+1e-12)),reduction_indices=shape(1))
  #KLD<--0.5*tf$reduce_sum(1+z_log_sigma_sq-tf$square(z_mean)-tf$exp(z_log_sigma_sq), reduction_indices=shape(1))
  #CVAE.cost=tf$reduce_mean(-likelihood+KLD)
  ###########


  log_var=sigma_weight*x_reconstr_logvar
  loss1=tf$reduce_mean(tf$multiply(tf$exp(-log_var),tf$reduce_sum(cost_list)))
 # loss1=tf$multiply(tf$exp(-log_var),tf$reduce_sum(cost_list))
  loss2=tf$reduce_mean(log_var)
  #loss2=tf$reduce_sum(log_var)
  joint_loss<-0.5*(loss1+loss2)




  #joint_loss<-tf$reduce_sum(cost_list)

  #KLD
   loss_latent<--0.5*tf$reduce_sum(1+z_log_sigma_sq-tf$square(z_mean)-tf$exp(z_log_sigma_sq), reduction_indices=shape(1))
  #loss_latent<-0.5*tf$reduce_sum(-1-z_log_sigma_sq+tf$square(z_mean)+tf$exp(z_log_sigma_sq), reduction_indices=shape(1))
  #loss_latent<-tf$maximum(-0.5*tf$reduce_mean(1+z_log_sigma_sq*tf$square(z_mean)-tf$exp(2-z_log_sigma_sq/2), reduction_indices=shape(1)),0)

  if(kld){
    #cost = tf$reduce_mean(loss_weight*tf$reduce_sum(cost_list)+kld_weight*loss_latent)
    cost = tf$reduce_mean(loss_weight*joint_loss+kld_weight*loss_latent)
  }else{
    cost = tf$reduce_mean(joint_loss)
  }


  return(cost)
}

#'midae cost function
midae_optimizer<-function(x,na_idx,networkOutput,output_split,output_struc){
  tf <- tensorflow::tf
  na_split=tf$split(na_idx,output_split,axis= as.integer(1))
  true_split=tf$split(x,output_split,axis= as.integer(1))
  x_reconstr_mean<-networkOutput$x_reconstr_mean
  x_reconstr_logvar<-networkOutput$x_reconstr_logvar
  pred_split=tf$split(x_reconstr_mean,output_split,axis=as.integer(1))
  cost_list=list()
  for (n in 1:length(output_struc)){
    na_adj=tf$cast(tf$count_nonzero(na_split[[n]]),tf$float32)/tf$cast(tf$size(na_split[[n]]),tf$float32)
    if (output_struc[n] == 'numeric'){

      #cost_list[[n]]=tf$losses$mean_squared_error(tf$boolean_mask(true_split[[n]],na_split[[n]]),tf$boolean_mask(pred_split[[n]],na_split[[n]]))*na_adj
      cost_list[[n]]=tf$sqrt(tf$losses$mean_squared_error(tf$boolean_mask(true_split[[n]],na_split[[n]]),tf$boolean_mask(pred_split[[n]],na_split[[n]])))*na_adj
      #cost_list[[n]]=tf$square(tf$boolean_mask(true_split[[n]],na_split[[n]])-tf$boolean_mask(pred_split[[n]],na_split[[n]]))*na_adj



      }
    else if (output_struc[n] == 'binary'){
      cost_list[[n]]=tf$losses$sigmoid_cross_entropy(tf$boolean_mask(true_split[[n]],na_split[[n]]),
                                                     tf$boolean_mask(pred_split[[n]],na_split[[n]]))*na_adj
    }
    else {
      cost_list[[n]]=tf$losses$softmax_cross_entropy(tf$reshape(tf$boolean_mask(true_split[[n]],na_split[[n]]),shape=shape(-1,output_struc[n])),
                                                     tf$reshape(tf$boolean_mask(pred_split[[n]],na_split[[n]]),shape=shape(-1,output_struc[n])))*na_adj
    }
  }



  #joint_loss<-tf$reduce_sum(cost_list)
  log_var=0.1*x_reconstr_logvar
  loss1=tf$reduce_mean(tf$multiply(tf$exp(-log_var),tf$reduce_sum(cost_list)))
  loss2=tf$reduce_mean(log_var)
  joint_loss<-0.5*(loss1+loss2)

  #log_var=tf$compat$v1$log(tf$nn$moments(x,axes=as.integer(1))[[2]])
  #loss1=tf$reduce_sum(tf$multiply(tf$exp(-log_var),tf$reduce_sum(cost_list)))
  #loss2=tf$reduce_mean(log_var)
  #joint_loss<-0.5*(loss1+loss2)

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

