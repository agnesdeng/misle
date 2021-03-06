#' Xavier Initialization using Uniform distribution

xavier_init<-function(n_inputs, n_outputs, constant=1){
  tf <- tensorflow::tf
  low = -constant*sqrt(6.0/(n_inputs + n_outputs))
  high = constant*sqrt(6.0/(n_inputs + n_outputs))
  return(tf$random_uniform(shape(n_inputs, n_outputs), minval=low, maxval=high, dtype=tf$float32))
}



#' Kaiming He Initialization
he_init<-function(n_inputs,n_outputs,constant=1){
  tf<-tensorflow::tf
  low = -constant*sqrt(3.0/(n_inputs))
  high = constant*sqrt(3.0/(n_inputs))
  return(tf$random_uniform(shape(n_inputs, n_outputs), minval=low, maxval=high, dtype=tf$float32))
}

#' Mivae initialisation
mivae_init<-function(encoder_structure,decoder_structure,
                     n_input, n_h){
  tf <- tensorflow::tf
  encoder_layer<-c(n_input,encoder_structure)
  decoder_layer<-c(decoder_structure,n_input)
  weights<-NULL
  ############################
  # Set-up Encoder
  ############################
  E=length(encoder_structure)
  for ( n in 1:E){
    weights[["encoder_weights"]][[n]]=tf$Variable(he_init(encoder_layer[n], encoder_layer[n+1]))
    weights[["encoder_biases"]][[n]]=tf$Variable(tf$zeros(shape(encoder_layer[n+1]), dtype=tf$float32))
  }


  ############################
  # Set-up Latent layer
  ############################
  weights[["encoder_weights"]][["out_mean"]]=tf$Variable(he_init(encoder_structure[E], n_h))
  weights[["encoder_weights"]][["out_log_sigma"]]=tf$Variable(he_init(encoder_structure[E], n_h))
  weights[["encoder_biases"]][["out_mean"]]=tf$Variable(tf$zeros(shape(n_h), dtype=tf$float32))
  weights[["encoder_biases"]][["out_log_sigma"]]=tf$Variable(tf$zeros(shape(n_h), dtype=tf$float32))

  ############################
  # Set-up Decoder
  ############################
  D=length(decoder_structure)
  weights[["decoder_weights"]][[1]]=tf$Variable(he_init(n_h, decoder_layer[1]))
  weights[["decoder_biases"]][[1]]=tf$Variable(tf$zeros(shape(decoder_layer[1]), dtype=tf$float32))

  if(D>1){
    for ( n in 2:D){
      weights[["decoder_weights"]][[n]]=tf$Variable(he_init(decoder_layer[n-1], decoder_layer[n]))
      weights[["decoder_biases"]][[n]]=tf$Variable(tf$zeros(shape(decoder_layer[n]), dtype=tf$float32))
    }
  }


  ############################
  # Set-up Output Layer
  ############################
  weights[["decoder_weights"]][["out_mean"]]=tf$Variable(he_init(decoder_layer[D], decoder_layer[D+1]))
  weights[["decoder_biases"]][["out_mean"]]=tf$Variable(tf$zeros(decoder_layer[D+1], dtype=tf$float32))
  weights[["decoder_weights"]][["out_logvar"]]=tf$Variable(he_init(decoder_layer[D], decoder_layer[D+1]))
  weights[["decoder_biases"]][["out_logvar"]]=tf$Variable(tf$zeros(decoder_layer[D+1], dtype=tf$float32))

  return(weights)
}


# Mivae Encoder update function
vae_encoder<-function(act,x, weights, biases,encoder_structure){
  tf <- tensorflow::tf
  E=length(encoder_structure)
  for(n in 1:E){
    x<-act(tf$add(tf$matmul(x, weights[[n]]), biases[[n]]))
  }

  z_mean = tf$add(tf$matmul(x, weights[['out_mean']]), biases[['out_mean']])
  z_log_sigma_sq = tf$add(tf$matmul(x, weights[['out_log_sigma']]), biases[['out_log_sigma']])
  return (list("z_mean"=z_mean, "z_log_sigma_sq"=z_log_sigma_sq))
}


# Mivae Decoder update function
vae_decoder<-function(act,z, weights, biases,decoder_structure){
  tf <- tensorflow::tf
  D=length(decoder_structure)
  for(n in 1:D){
    z<-act(tf$add(tf$matmul(z, weights[[n]]), biases[[n]]))
  }
   #x_reconstr_mean<-tf$nn$sigmoid(tf$add(tf$matmul(z, weights[['out_mean']]), biases[['out_mean']]))
   #x_reconstr_logvar<-tf$nn$sigmoid(tf$add(tf$matmul(z, weights[['out_logvar']]), biases[['out_logvar']]))
   #x_reconstr_mean<-tf$clip_by_value(tf$add(tf$matmul(z, weights[['out_mean']]), biases[['out_mean']]),0,1)
   x_reconstr_mean<-tf$add(tf$matmul(z, weights[['out_mean']]), biases[['out_mean']])
   x_reconstr_logvar<-tf$add(tf$matmul(z, weights[['out_logvar']]), biases[['out_logvar']])


   #x_reconstr_mean<-tf$add(tf$matmul(z, weights[['out_mean']]), biases[['out_mean']])
  #return(x_reconstr_mean)
   return(list("x_reconstr_mean"=x_reconstr_mean, "x_reconstr_logvar"=x_reconstr_logvar))
}

#' Mivae output evaluation
mivae_output<-function(act,x,network_weights,feed_size,n_h,encoder_structure,decoder_structure){
  tf <- tensorflow::tf

  LatentParameter<-vae_encoder(act,x, network_weights[["encoder_weights"]], network_weights[["encoder_biases"]],encoder_structure)
  z_mean<-LatentParameter$z_mean
  z_log_sigma_sq <-LatentParameter$z_log_sigma_sq

  # Draw one sample z from Gaussian distribution
  #feed_size=batch_size
  #eps = tf$random_normal(shape(feed_size, n_h), 0, tf$sqrt(tf$exp(z_log_sigma_sq)), dtype=tf$float32)
  eps = tf$random_normal(shape(feed_size, n_h), 0, 1, dtype=tf$float32)
  #eps = tf$random_normal(shape(n_h), 0, 1, dtype=tf$float32)
  # z = mu + sigma*epsilon
  #z = tf$add(z_mean, tf$multiply(tf$sqrt(tf$exp(z_log_sigma_sq)), eps))
  z = tf$add(z_mean, tf$multiply(tf$exp(z_log_sigma_sq/2), eps))

  Out<- vae_decoder(act,z, network_weights[["decoder_weights"]], network_weights[["decoder_biases"]],decoder_structure)
  #x_reconstr_mean<-Out$x_reconstr_logvar
  x_reconstr_mean<-Out$x_reconstr_mean
  x_reconstr_logvar<-Out$x_reconstr_logvar
  return(list("x_reconstr_mean"=x_reconstr_mean, "x_reconstr_logvar"=x_reconstr_logvar,"z_log_sigma_sq"=z_log_sigma_sq, "z_mean"=z_mean,"z"=z))
}


