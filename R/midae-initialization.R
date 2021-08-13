#' Xavier Initialization using Uniform distribution
#' @export
xavier_init<-function(n_inputs, n_outputs, constant=1){
  tf <- tensorflow::tf
  low = -constant*sqrt(6.0/(n_inputs + n_outputs))
  high = constant*sqrt(6.0/(n_inputs + n_outputs))
  return(tf$random$uniform(shape(n_inputs, n_outputs), minval=low, maxval=high, dtype=tf$float32))
}



#' Kaiming He Initialization
#' @export
he_init<-function(n_inputs,n_outputs,constant=1){
  tf<-tensorflow::tf
  low = -constant*sqrt(3.0/(n_inputs))
  high = constant*sqrt(3.0/(n_inputs))
  return(tf$random$uniform(shape(n_inputs, n_outputs), minval=low, maxval=high, dtype=tf$float32))
}


#'Midae initialisation
midae_init<-function(encoder_structure,decoder_structure,
                     n_input, n_h){
  tf <- tensorflow::tf
  tf$compat$v1$disable_eager_execution()
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
  weights[["encoder_biases"]][["out_mean"]]=tf$Variable(tf$zeros(shape(n_h), dtype=tf$float32))


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




#' Midae Encoder update function
denoise_encoder<-function(act,x,weights, biases,input_drop,hidden_drop,encoder_structure){
  tf <- tensorflow::tf
  tf$compat$v1$disable_eager_execution()
  x<-act(tf$add(tf$matmul(tf$nn$dropout(x,input_drop), weights[[1]]), biases[[1]]))
  E=length(encoder_structure)
  if(E>1){
    for(n in 2:E){
      x<-act(tf$add(tf$matmul(tf$nn$dropout(x,hidden_drop), weights[[n]]), biases[[n]]))
    }
  }
  z_mean = act(tf$add(tf$matmul(x, weights[['out_mean']]), biases[['out_mean']]))
  return (z_mean)
}




#' Midae Decoder update function
denoise_decoder<-function(act,z, weights, biases, hidden_drop,decoder_structure){
  tf <- tensorflow::tf
  tf$compat$v1$disable_eager_execution()
  D=length(decoder_structure)
  for(n in 1:D){
    z<-act(tf$add(tf$matmul(tf$nn$dropout(z,hidden_drop), weights[[n]]), biases[[n]]))
  }

  x_reconstr_mean<-tf$add(tf$matmul(tf$nn$dropout(z,hidden_drop), weights[['out_mean']]), biases[['out_mean']])
  #x_reconstr_logvar<-tf$nn$sigmoid(tf$add(tf$matmul(z, weights[['out_logvar']]), biases[['out_logvar']]))
  x_reconstr_logvar<-tf$add(tf$matmul(tf$nn$dropout(z,hidden_drop), weights[['out_logvar']]), biases[['out_logvar']])
  #x_reconstr_mean<-tf$add(tf$matmul(z, weights[['out_mean']]), biases[['out_mean']])
  #return(x_reconstr_mean)
  return(list("x_reconstr_mean"=x_reconstr_mean, "x_reconstr_logvar"=x_reconstr_logvar))

  return(x_reconstr_mean)
}


#' Midae output evaluation
midae_output<-function(act,x,network_weights,encoder_structure,decoder_structure,input_drop,hidden_drop){
  tf <- tensorflow::tf
  tf$compat$v1$disable_eager_execution()
  #compressed value
  z<-denoise_encoder(act,x, network_weights[["encoder_weights"]], network_weights[["encoder_biases"]],input_drop,hidden_drop,encoder_structure)
  Out <-denoise_decoder(act,z, network_weights[["decoder_weights"]],network_weights[["decoder_biases"]], hidden_drop,decoder_structure)
  x_reconstr_mean<-Out$x_reconstr_mean
  x_reconstr_logvar<-Out$x_reconstr_logvar
  return(list("x_reconstr_mean"=x_reconstr_mean, "x_reconstr_logvar"=x_reconstr_logvar))
}
