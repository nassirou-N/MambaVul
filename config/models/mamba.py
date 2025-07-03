from __future__ import print_function
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 = INFO, 2 = WARNING, 3 = ERROR)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow logging
from config.arg_parser import parameter_parser
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input, Dropout, LayerNormalization, Activation
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np

# Set print options to display the entire array
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

args = parameter_parser()

class MambaBlock(Layer):
    """
    Simplified Mamba block implementation for sequence modeling
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2, dropout_rate=0.1, **kwargs):
        super(MambaBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = expand_factor * d_model
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # Input projection
        self.in_proj = Dense(self.d_inner * 2, use_bias=False)
        
        # Convolution layer
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.d_inner,
            kernel_size=self.d_conv,
            padding='same',
            groups=self.d_inner,
            use_bias=True
        )
        
        # Activation
        self.activation = Activation('silu')
        
        # SSM parameters
        self.A = self.add_weight(
            name='A',
            shape=(self.d_inner, self.d_state),
            initializer='random_normal',
            trainable=True
        )
        
        self.B = Dense(self.d_state, use_bias=False)
        self.C = Dense(self.d_state, use_bias=False)
        self.D = self.add_weight(
            name='D',
            shape=(self.d_inner,),
            initializer='ones',
            trainable=True
        )
        
        # Output projection
        self.out_proj = Dense(self.d_model, use_bias=False)
        
        # Dropout
        self.dropout = Dropout(self.dropout_rate)
        
        super(MambaBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        batch_size, seq_len, d_model = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        
        # Input projection
        x_and_res = self.in_proj(inputs)
        x, res = tf.split(x_and_res, 2, axis=-1)
        
        # Convolution
        x = self.conv1d(x)
        x = self.activation(x)
        
        # SSM computation (simplified)
        B = self.B(x)  # (batch, seq_len, d_state)
        C = self.C(x)  # (batch, seq_len, d_state)
        
        # Simplified state space computation
        # In practice, this would involve more complex recurrent computation
        # Here we use a simplified version for demonstration
        A_expanded = tf.expand_dims(tf.expand_dims(self.A, 0), 0)  # (1, 1, d_inner, d_state)
        
        # Simplified SSM: y = Cx + Dx (without proper state evolution)
        # This is a simplified version - full Mamba would have proper selective scan
        y = tf.einsum('bld,bls->bls', x, C) + tf.expand_dims(self.D, 0) * x
        
        # Gating with residual
        y = y * tf.nn.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        # Dropout
        output = self.dropout(output, training=training)
        
        return output
    
    def get_config(self):
        config = super(MambaBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_state': self.d_state,
            'd_conv': self.d_conv,
            'expand_factor': self.expand_factor,
            'dropout_rate': self.dropout_rate
        })
        return config

class MAMBA_Model:
    def __init__(self, data, args):
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.dropout = args.dropout
        
        # Prepare data
        self.vectors = np.stack(data.iloc[:, 0].values)
        self.labels = data.iloc[:, 1].values
        
        positive_idxs = np.where(self.labels == 1)[0]
        negative_idxs = np.where(self.labels == 0)[0]
        idxs = np.concatenate([positive_idxs, negative_idxs])
        
        x_train, x_test, y_train, y_test = train_test_split(
            self.vectors[idxs], self.labels[idxs],
            test_size=0.2, stratify=self.labels[idxs], random_state=42
        )
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        
        # Class weights for imbalanced dataset
        classes = np.array([0, 1])
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.labels)
        self.class_weight = {index: weight for index, weight in enumerate(class_weights)}
        
        # Build model
        input_shape = (self.x_train.shape[1], self.x_train.shape[2])
        self.model = self.build_model(input_shape)
    
    def build_model(self, input_shape):
        inputs = Input(shape=input_shape)
        
        # Input normalization
        x = LayerNormalization()(inputs)
        
        # Mamba blocks
        x = MambaBlock(
            d_model=input_shape[-1],
            d_state=16,
            d_conv=4,
            expand_factor=2,
            dropout_rate=self.dropout
        )(x)
        
        # Additional Mamba block for deeper processing
        x = MambaBlock(
            d_model=input_shape[-1],
            d_state=16,
            d_conv=4,
            expand_factor=2,
            dropout_rate=self.dropout
        )(x)
        
        # Global pooling to reduce sequence dimension
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        outputs = Dense(2, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(learning_rate=self.lr)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self):
        print("Training Mamba model...")
        self.model.fit(
            self.x_train, self.y_train,
            epochs=self.epochs,
            class_weight=self.class_weight,
            verbose=1,
            batch_size=self.batch_size,
            validation_split=0.1
        )
    
    def test(self):
        print("Testing Mamba model...")
        values = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size, verbose=0)
        print(f"\nAccuracy: {values[1]:.4f}")
        
        predictions = self.model.predict(self.x_test, batch_size=self.batch_size, verbose=0)
        predictions = np.round(predictions)
        
        tn, fp, fn, tp = confusion_matrix(
            np.argmax(self.y_test, axis=1),
            np.argmax(predictions, axis=1)
        ).ravel()
        
        print(f'False positive rate(FP): {fp / (fp + tn):.4f}')
        print(f'False negative rate(FN): {fn / (fn + tp):.4f}')
        
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1_score = (2 * precision * recall) / (precision + recall)
        
        print(f'Recall: {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'F1 score: {f1_score:.4f}')
        
        return {
            'accuracy': values[1],
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }