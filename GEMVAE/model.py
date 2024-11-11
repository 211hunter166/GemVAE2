import tensorflow.compat.v1 as tf
import tensorflow as tf
import tensorflow.compat.v1 as v1
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

class LinBnDrop(tf.keras.Sequential):
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=True):
        layers = []
        if bn:
            layers.append(tf.keras.layers.BatchNormalization())
        if p != 0:
            layers.append(tf.keras.layers.Dropout(p))
        lin = [tf.keras.layers.Dense(n_out, use_bias=not bn)]
        if act is not None:
            layers.append(act)
        layers = lin + layers if lin_first else layers + lin
        super(LinBnDrop, self).__init__(layers)


class GATE():



    @staticmethod
    def contrastive_loss_function(y_true, y_pred, margin=1.0):
        """
        Calculates contrastive loss, aiming to bring positive pairs closer 
        and push negative pairs apart.
        """
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    

    def encode_all_layers(self, A, prune_A, data, is_gene_modality):
        """Encodes data through all layers, either using gene or protein modality."""
        H = data
        n_layers = self.n_layers1 if is_gene_modality else self.n_layers2
        for layer in range(n_layers):
            if is_gene_modality:
                H = self.__encoder1(A, prune_A, H, layer)
            else:
                H = self.__encoder2(A, prune_A, H, layer)
            if self.nonlinear and layer != n_layers - 1:
                H = tf.nn.elu(H)
        return H

    @staticmethod
    @tf.function  # Decorate with @tf.function for graph execution
    def project_embedding(neighbor_embedding, target_dim):
        projection_layer = tf.keras.layers.Dense(target_dim)
        return projection_layer(neighbor_embedding)

    @staticmethod
    @tf.function  # Ensures the function runs in graph mode
    def create_pairs(embedding, neighbors, encoder_model, original_data, A, prune_A, is_gene_modality):
        """
        Create positive and negative pairs for contrastive learning.
        Positive pairs are node embedding and local neighbor representations.
        Negative pairs are created by shuffling data, passing through encoder, and then finding neighbors.
        """
        # Convert neighbors (a coo_matrix) to a SparseTensor
        neighbors = tf.sparse.SparseTensor(
            indices=tf.convert_to_tensor(list(zip(neighbors.row, neighbors.col)), dtype=tf.int64),
            values=neighbors.data,
            dense_shape=neighbors.shape
        )

        # Helper function to create a positive pair
        def create_positive_pair(i):
            neighbor_indices = tf.cast(tf.sparse.to_dense(tf.sparse.slice(neighbors, [i, 0], [1, neighbors.shape[1]])), tf.int32)
            neighbor_embedding = tf.reduce_sum(tf.gather(embedding, neighbor_indices), axis=0)
            return (embedding[i], neighbor_embedding)

        positive_pairs = tf.map_fn(create_positive_pair, tf.range(tf.shape(embedding)[0]), dtype=(embedding.dtype, embedding.dtype))

        # Shuffle original data to create corrupted (negative) samples
        corrupted_data = tf.random.shuffle(original_data)
        
        # Encode the corrupted data
        corrupted_embeddings = encoder_model.encode_all_layers(A, prune_A, corrupted_data, is_gene_modality)

        def create_negative_pair(i):
            neighbor_indices = tf.cast(tf.sparse.to_dense(tf.sparse.slice(neighbors, [i, 0], [1, neighbors.shape[1]])), tf.int32)
            corrupted_neighbor_embedding = tf.reduce_sum(tf.gather(corrupted_embeddings, neighbor_indices), axis=0)
            return (embedding[i], corrupted_neighbor_embedding)

        negative_pairs = tf.map_fn(create_negative_pair, tf.range(tf.shape(embedding)[0]), dtype=(embedding.dtype, embedding.dtype))

        return positive_pairs, negative_pairs



    def __init__(self, hidden_dims1, hidden_dims2,z_dim=30,alpha=0.3, nonlinear=True, weight_decay=0.0001, num_hidden=256, num_proj_hidden=256, tau=0.5,kl_loss = 0.02,contrastive_loss = 0.1,recon_loss = 1,weight_decay_loss = 1,recon_loss_type = "MSE"):
        self.n_layers1 = len(hidden_dims1) - 1
        self.n_layers2 = len(hidden_dims2) - 1
        self.alpha = alpha
        self.W1, self.v1, self.prune_v1 = self.define_weights1(hidden_dims1, self.n_layers1)
        self.W2, self.v2, self.prune_v2 = self.define_weights2(hidden_dims2, self.n_layers2)
        self.C1 = {}
        self.C2 = {}
        self.prune_C1 = {}
        self.prune_C2 = {}
        self.nonlinear = nonlinear
        self.weight_decay = weight_decay
        self.z_dim = z_dim
        self.fc_mu = LinBnDrop(z_dim,z_dim, p=0.2)
        self.fc_var = LinBnDrop(z_dim,z_dim, p=0.2)

        self.tau = tau
        self.fc1 = tf.keras.layers.Dense(num_proj_hidden, activation='elu')
        self.fc2 = tf.keras.layers.Dense(num_hidden)
        self.dropout_rate = 0.1

        self.kl_loss = kl_loss
        self.contrastive_loss = contrastive_loss
        self.recon_loss = recon_loss
        self.weight_decay_loss = weight_decay_loss
        self.recon_loss_type = recon_loss_type

        # Decoder 1
        self.W_dec1 = {}
        for layer in range(self.n_layers1 - 1, -1, -1):
            self.W_dec1[layer] = tf.Variable(tf.random.normal([hidden_dims1[layer+1], hidden_dims1[layer]]))

        # Decoder 2
        self.W_dec2 = {}
        for layer in range(self.n_layers2 - 1, -1, -1):
            self.W_dec2[layer] = tf.Variable(tf.random.normal([hidden_dims2[layer+1], hidden_dims2[layer]]))
    
    def convert_coo_to_sparse_tensor(coo_matrix):
        """Convert a scipy.sparse.coo_matrix to tf.SparseTensor."""
        indices = np.vstack((coo_matrix.row, coo_matrix.col)).T
        values = coo_matrix.data
        dense_shape = coo_matrix.shape
        return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)


    def __call__(self, A1,A2 ,prune_A1,prune_A2, X1,X2,G1,G2):
        # Encoder 1
        H1 = X1
        for layer in range(self.n_layers1):
            H1 = self.__encoder1(A1, prune_A1, H1, layer)
            if self.nonlinear:
                if layer != self.n_layers1 - 1:
                    H1 = tf.nn.elu(H1)

        # Encoder 2
        H2 = X2
        for layer in range(self.n_layers2):
            H2 = self.__encoder2(A2, prune_A2, H2, layer)
            if self.nonlinear:
                if layer != self.n_layers2 - 1:
                    H2 = tf.nn.elu(H2)

        con_loss = self.con_loss(H1, H2, batch_size=0)

        self.c_loss = con_loss

        # Concatenate encoder outputs
        H = tf.concat([H1, H2], axis=1)
        
        # Call the third encoder
        global latent_rep 
        H = self.__encoder3(H)
        print(H.shape)

        if self.kl_loss != 0:
            #Latent space using a varational auto encoder
            mu = self.fc_mu(H)
            var = self.fc_var(H)
            H = self.reparameterize(mu, var)
            
            

            # KL Divergence Loss
            kl_divergance_loss = -0.5 * tf.reduce_sum(1 + var - tf.square(mu) - tf.exp(var), axis=1)
            kl_divergance_loss = tf.reduce_mean(kl_divergance_loss)
        else:
            kl_divergance_loss=0

        H1,H2=H,H
        latent_rep = H

        # Decoder 1
        for layer in range(self.n_layers1 - 1, -1, -1):
            H1 = self.__decoder1(H1, layer)
            if self.nonlinear:
                if layer != 0:
                    H1 = tf.nn.elu(H1)
        X1_ = H1


        # Decoder 2
        for layer1 in range(self.n_layers2 - 1, -1, -1):
            H2 = self.__decoder2(H2, layer1)
            if self.nonlinear:
                if layer1 != 0:
                    H2 = tf.nn.elu(H2)
        X2_ = H2

        # Loss calculation
        # Calculating inputs for the ZINB loss
        # Data normalization (optional)
        X1_, X2_ = v1.nn.softmax(X1_), v1.nn.softmax(X2_)

        if self.recon_loss_type == 'ZINB':
            #USING ZINB FOR LOSS CALC
            # Estimate library size as in reference code
            log_library_size1 = v1.math.log(v1.reduce_sum(X1_, axis=-1) + 1)
            #log_library_size2 = v1.math.log(v1.reduce_sum(X2_, axis=-1) + 1)

            library_size_mean1 = v1.reduce_mean(log_library_size1)
            #library_size_variance1 = v1.math.reduce_variance(log_library_size1)

            #library_size_mean2 = v1.reduce_mean(log_library_size2)
            #library_size_variance2 = v1.math.reduce_variance(log_library_size2)

            self.x_post_r1 = v1.random.normal(shape=[X1_.shape[-1]], dtype=v1.float32)
            #self.x_post_r2 = v1.random.normal(shape=[X2_.shape[-1]], dtype=v1.float32)

            # They used an additional layer between decoder and zinb loss
            # You can consider adding it if the performance is not satisfactory 

            x_post_scale1 = v1.exp(library_size_mean1) * X1_
            #x_post_scale2 = v1.exp(library_size_mean2) * X2_

            local_dispersion1 = v1.exp(self.x_post_r1)
            #local_dispersion2 = v1.exp(self.x_post_r2)

            x_post_dropout1 = v1.nn.dropout(X1_, self.dropout_rate)
            #x_post_dropout2 = v1.nn.dropout(X2_, self.dropout_rate)

            # ZINB Loss calculation
            zinb_loss1 = self.zinb_model(X1, x_post_scale1, local_dispersion1, x_post_dropout1)
            #zinb_loss2 = self.zinb_model(X2, x_post_scale2, local_dispersion2, x_post_dropout2)
            
            # Calculate the mean of zinb_loss1 and reconstruction_loss

            
            rloss = tf.reduce_mean(zinb_loss1) 
            #rloss += tf.reduce_mean(zinb_loss2)
            rloss*=-0.5

            
        else:
            #using MSE
            print("Using MSE for gene")
            rloss = tf.sqrt(tf.reduce_sum(tf.pow(X1 - X1_, 2)))
            
        #MSE always for protien 
        rloss += tf.sqrt(tf.reduce_sum(tf.pow(X2 - X2_, 2)))
  



        weight_decay_loss = 0
        for layer in range(self.n_layers1):
            weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W1[layer]), self.weight_decay, name='weight_loss')
        for layer in range(self.n_layers2):
            weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W2[layer]), self.weight_decay, name='weight_loss')
        for layer in range(self.n_layers1):
            weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W_dec1[layer]), self.weight_decay, name='weight_loss')
        for layer in range(self.n_layers2):
            weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W_dec2[layer]), self.weight_decay, name='weight_loss')

        # Total loss
        print("Loss weights are = ",self.contrastive_loss,self.recon_loss,self.weight_decay_loss,self.kl_loss)
        
        # Calculate positive and negative pairs and contrastive loss
        
        # Convert your neighbors matrices to TensorFlow SparseTensor format
        G1_sparse = self.convert_coo_to_sparse_tensor(G1)
        G2_sparse = self.convert_coo_to_sparse_tensor(G2)

        pos_pairs1, neg_pairs1 = self.create_pairs(
            embedding=H1,
            neighbors=G1_sparse,  # Use the converted sparse tensor here
            encoder_model=self,
            original_data=X1,
            A=A1,
            prune_A=prune_A1,
            is_gene_modality=True
        )

        # Project the neighbor embeddings in each pair for gene modality
        projected_pos_pairs1 = tf.map_fn(
            lambda pair: (pair[0], self.project_embedding(pair[1], target_dim=H1.shape[-1])),
            pos_pairs1,
            fn_output_signature=(H1.dtype, H1.dtype)
        )

        projected_neg_pairs1 = tf.map_fn(
            lambda pair: (pair[0], self.project_embedding(pair[1], target_dim=H1.shape[-1])),
            neg_pairs1,
            fn_output_signature=(H1.dtype, H1.dtype)
        )

        # Calculate contrastive loss for projected positive and negative pairs
        contrastive_loss1 = tf.reduce_sum(
            tf.map_fn(lambda pair: self.contrastive_loss_function(1, tf.norm(pair[0] - pair[1])),
                    projected_pos_pairs1, fn_output_signature=tf.float32)
        ) + tf.reduce_sum(
            tf.map_fn(lambda pair: self.contrastive_loss_function(0, tf.norm(pair[0] - pair[1])),
                    projected_neg_pairs1, fn_output_signature=tf.float32)
        )

        # Repeat for protein modality (H2 and G2)
        pos_pairs2, neg_pairs2 = self.create_pairs(
            embedding=H2,
            neighbors=G2_sparse,  # Use the converted sparse tensor here
            encoder_model=self,
            original_data=X2,
            A=A2,
            prune_A=prune_A2,
            is_gene_modality=False
        )

        # Project the neighbor embeddings in each pair for protein modality
        projected_pos_pairs2 = tf.map_fn(
            lambda pair: (pair[0], self.project_embedding(pair[1], target_dim=H2.shape[-1])),
            pos_pairs2,
            fn_output_signature=(H2.dtype, H2.dtype)
        )

        projected_neg_pairs2 = tf.map_fn(
            lambda pair: (pair[0], self.project_embedding(pair[1], target_dim=H2.shape[-1])),
            neg_pairs2,
            fn_output_signature=(H2.dtype, H2.dtype)
        )

        # Calculate contrastive loss for projected positive and negative pairs for protein modality
        contrastive_loss2 = tf.reduce_sum(
            tf.map_fn(lambda pair: self.contrastive_loss_function(1, tf.norm(pair[0] - pair[1])),
                    projected_pos_pairs2, fn_output_signature=tf.float32)
        ) + tf.reduce_sum(
            tf.map_fn(lambda pair: self.contrastive_loss_function(0, tf.norm(pair[0] - pair[1])),
                    projected_neg_pairs2, fn_output_signature=tf.float32)
        )

        # Total contrastive loss
        total_contrastive_loss = contrastive_loss1 + contrastive_loss2

    
        # Add weighted contrastive loss to the total loss
        self.loss = (self.contrastive_loss * total_contrastive_loss) + (self.recon_loss * rloss) +                 (self.weight_decay_loss * weight_decay_loss) + (self.kl_loss * kl_divergance_loss)
    

        if self.alpha == 0:
            print("\n\nAlpha = 0")
            self.Att_l = {'C1': self.C1, 'C2': self.C2}
        else:
            self.Att_l = {'C1': self.C1, 'C2': self.C2, 'prune_C1': self.prune_C1, 'prune_C2': self.prune_C2}
            

        return self.c_loss, self.loss, latent_rep, self.Att_l, X1_, X2_

    
    # Define the zinb_model loss function
    def zinb_model(self, x, mean, inverse_dispersion, logit, eps=1e-4):
        expr_non_zero = - v1.nn.softplus(- logit) \
                        + v1.log(inverse_dispersion + eps) * inverse_dispersion \
                        - v1.log(inverse_dispersion + mean + eps) * inverse_dispersion \
                        - x * v1.log(inverse_dispersion + mean + eps) \
                        + x * v1.log(mean + eps) \
                        - v1.lgamma(x + 1) \
                        + v1.lgamma(x + inverse_dispersion) \
                        - v1.lgamma(inverse_dispersion) \
                        - logit 
        
        expr_zero = - v1.nn.softplus( - logit) \
                    + v1.nn.softplus(- logit + v1.log(inverse_dispersion + eps) * inverse_dispersion \
                                    - v1.log(inverse_dispersion + mean + eps) * inverse_dispersion) 
        
        template = v1.cast(v1.less(x, eps), v1.float32)
        expr =  v1.multiply(template, expr_zero) + v1.multiply(1 - template, expr_non_zero)
        
        return v1.reduce_sum(expr, axis=-1)
    
    def projection(self, z):
        z = self.fc1(z)
        return self.fc2(z)

    def sim(self, z1, z2):
        z1 = tf.nn.l2_normalize(z1, axis=1)
        z2 = tf.nn.l2_normalize(z2, axis=1)
        return tf.matmul(z1, z2, transpose_b=True)

    def semi_loss(self, z1, z2):
        f = lambda x: tf.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        diag_ref_sim = tf.linalg.diag_part(refl_sim)

        return -tf.math.log(
            tf.linalg.diag_part(between_sim)
            / (tf.reduce_sum(refl_sim, axis=1) + tf.reduce_sum(between_sim, axis=1) - diag_ref_sim))

    def batched_semi_loss(self, z1, z2, batch_size):
        num_nodes = tf.shape(z1)[0]
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: tf.exp(x / self.tau)
        indices = tf.range(0, num_nodes)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(tf.gather(z1, mask), z1))  # [B, N]
            between_sim = f(self.sim(tf.gather(z1, mask), z2))  # [B, N]

            losses.append(-tf.math.log(
                tf.linalg.diag_part(tf.gather(between_sim, mask, batch_dims=1))
                / (tf.reduce_sum(refl_sim, axis=1) + tf.reduce_sum(between_sim, axis=1)
                   - tf.linalg.diag_part(tf.gather(refl_sim, mask, batch_dims=1)))))

        return tf.concat(losses, axis=0)

    def con_loss(self, z1, z2, mean=True, batch_size=0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = tf.reduce_mean(ret) if mean else tf.reduce_sum(ret)

        return ret
    

    def reparameterize(self, mu, logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(std))
        return eps * std + mu

    def __encoder1(self, A, prune_A1, H, layer):
        ##print('enc1 = ',H)
        H = tf.matmul(H, self.W1[layer])
        if layer == self.n_layers1 - 1:
            return H
        self.C1[layer] = self.graph_attention_layer(A, H, self.v1[layer], layer)
        if self.alpha == 0:
            return tf.sparse.sparse_dense_matmul(self.C1[layer], H)
        else:
            self.prune_C1[layer] = self.graph_attention_layer(prune_A1, H, self.prune_v1[layer], layer)
            return (1 - self.alpha) * tf.sparse.sparse_dense_matmul(self.C1[layer], H) + self.alpha * tf.sparse.sparse_dense_matmul(
                self.prune_C1[layer], H)
        
        

    def __encoder2(self, A, prune_A2, H, layer):
        #print('enc2 = ',H)
        H = tf.matmul(H, self.W2[layer])
        if layer == self.n_layers2 - 1:
            return H
        self.C2[layer] = self.graph_attention_layer(A, H, self.v2[layer], layer)
        if self.alpha == 0:
            return tf.sparse.sparse_dense_matmul(self.C2[layer], H)
        else:
            self.prune_C2[layer] = self.graph_attention_layer(prune_A2, H, self.prune_v2[layer], layer)
            return (1 - self.alpha) * tf.sparse.sparse_dense_matmul(self.C2[layer], H) + self.alpha * tf.sparse.sparse_dense_matmul(
                self.prune_C2[layer], H)
    
    def __decoder1(self, H, layer):
        #print('dec1 = ',H)
        H = tf.matmul(H, self.W1[layer], transpose_b=True)
        if layer == 0:

            return H
        if self.alpha == 0:

            return tf.sparse.sparse_dense_matmul(self.C1[layer-1], H)
        else:

            return (1 - self.alpha) * tf.sparse.sparse_dense_matmul(self.C1[layer-1], H) + self.alpha * tf.sparse.sparse_dense_matmul(
                self.prune_C1[layer-1], H)
        
    def __decoder2(self, H, layer):
        #print('dec2 = ',H)
        H = tf.matmul(H, self.W2[layer], transpose_b=True)
        if layer == 0:

            return H
        if self.alpha == 0:
            return tf.sparse.sparse_dense_matmul(self.C2[layer-1], H)
        
        else:

            return (1 - self.alpha) * tf.sparse.sparse_dense_matmul(self.C2[layer-1], H) + self.alpha * tf.sparse.sparse_dense_matmul(
                self.prune_C2[layer-1], H)
        

    def __encoder3(self, H):
        #print('enc3 = ',H)
        H = tf.keras.layers.Dense(self.z_dim)(H)
        #print('LATENT = ',H)
        return H


    

    def define_weights1(self,hidden_dims,n_layers):
        W = {}
        ##print('TOTAL LEYRS = ',n_layers)
        #n_layers=len(n_layers)-1
        #print('n_layers gene = ',n_layers)
        #print('Hidden dim gene = ',hidden_dims)

        for i in range(n_layers):
            W[i] = v1.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i+1]))

        # # Similar initialization for W_dec1
        # W_dec1 = {}
        # for i in range(n_layers - 1, -1, -1):
        #     W_dec1[i] = v1.get_variable("W_dec%s" % i, shape=(hidden_dims[i+1], hidden_dims[i]))  # Note the reversed order of hidden_dims
        
        Ws_att = {}
        for i in range(n_layers-1):
            V= {}
            V[0] = v1.get_variable("V%s_0" % i, shape=(hidden_dims[i+1], 1))
            V[1] = v1.get_variable("V%s_1" % i, shape=(hidden_dims[i+1], 1))

            Ws_att[i] = V
        if self.alpha == 0:
            return W, Ws_att, None
        prune_Ws_att = {}
        for i in range(n_layers-1):
            prune_V = {}
            prune_V[0] = v1.get_variable("prune_V%s_0" % i, shape=(hidden_dims[i+1], 1))
            prune_V[1] = v1.get_variable("prune_V%s_1" % i, shape=(hidden_dims[i+1], 1))

            prune_Ws_att[i] = prune_V

        return W, Ws_att, prune_Ws_att
    
    def define_weights2(self,hidden_dims,n_layers):
        w = {}
        ##print('TOTAL LEYRS = ',n_layers)
        #n_layers=len(n_layers)-1
        #print('n_layers protein = ',n_layers)
        #print('Hidden dim protein = ',hidden_dims)

        for i in range(n_layers):
            w[i] = v1.get_variable("w%s" % i, shape=(hidden_dims[i], hidden_dims[i+1]))
        
        # # Similar initialization for W_dec1
        # W_dec2 = {}
        # for i in range(n_layers - 1, -1, -1):
        #     W_dec2[i] = v1.get_variable("W_dec%s" % i, shape=(hidden_dims[i+1], hidden_dims[i]))  # Note the reversed order of hidden_dims
        
        ws_att = {}
        for i in range(n_layers-1):
            v = {}
            v[0] = v1.get_variable("v%s_0" % i, shape=(hidden_dims[i+1], 1))
            v[1] = v1.get_variable("v%s_1" % i, shape=(hidden_dims[i+1], 1))

            ws_att[i] = v
        if self.alpha == 0:
            return w, ws_att, None
        prune_ws_att = {}
        for i in range(n_layers-1):
            prune_v = {}
            prune_v[0] = v1.get_variable("prune_v%s_0" % i, shape=(hidden_dims[i+1], 1))
            prune_v[1] = v1.get_variable("prune_v%s_1" % i, shape=(hidden_dims[i+1], 1))

            prune_ws_att[i] = prune_v

        return w, ws_att, prune_ws_att
    


    def graph_attention_layer(self, A, M, v, layer):

        with v1.variable_scope("layer_%s" % layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            f2 = A * tf.transpose(f2, [1, 0])
            logits = v1.sparse_add(f1, f2)

            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                         values=tf.nn.sigmoid(logits.values),
                                         dense_shape=logits.dense_shape)
            attentions = v1.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions