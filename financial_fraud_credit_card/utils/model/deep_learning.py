import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow import keras


def deeplearning_model(X, y, X_valid, y_valid, _verbose=1, epochs=5000, batch_size=512, learning_rate=0.001):    
    input_shape = [X.shape[1]]
    
    model = keras.Sequential([
        layers.BatchNormalization(input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)),    
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),        
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),        
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)),        
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.01)),        
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(2, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    
    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['binary_accuracy', 
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc")
            ]
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        patience=20,
        min_delta=0.001,
        restore_best_weights=True,
    )

    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        patience=5,  # Increased patience
        factor=0.2,
        min_lr=1e-6,  # Lower min_lr for better optimization
    )
    
    history = model.fit(
        X, y,
        validation_data=(X_valid, y_valid),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, lr_schedule],
        verbose=_verbose
    )
    
    history_df = pd.DataFrame(history.history)
    
    history_df[['loss', 'val_loss', 'precision', 'recall', 'auc']].plot(title="Cross-entropy Loss")
    plt.show()
    
    history_df[['binary_accuracy', 'val_binary_accuracy', 'val_precision', 'val_recall', 'val_auc', 'lr' ]].plot(title="Binary Accuracy")
    plt.show()

    print(("Best Validation Loss: {:0.4f}" +
           "\nBest Validation Accuracy: {:0.4f}" +
           "\nBest Validation Precision: {:0.4f}" + 
           "\nBest Validation Recall: {:0.4f}" + 
           "\nBest Validation AUC: {:0.4f}")
          .format(history_df['val_loss'].min(), 
                  history_df['val_binary_accuracy'].max(),
                  history_df['val_precision'].max(),
                  history_df['val_recall'].max(),
                  history_df['val_auc'].max()))
    
    return model