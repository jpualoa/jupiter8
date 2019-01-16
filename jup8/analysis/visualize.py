import matplotlib.pyplot as plt

def plot_history(history):
    fig, axes = plt.subplots(2, 1)
    acc, loss = axes

    # summarize history for accuracy
    acc.plot(history.history['acc'])
    acc.plot(history.history['val_acc'])
    acc.title('model accuracy')
    acc.ylabel('accuracy')
    acc.xlabel('epoch')
    acc.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    loss.plot(history.history['loss'])
    loss.plot(history.history['val_loss'])
    loss.title('model loss')
    loss.ylabel('loss')
    loss.xlabel('epoch')
    loss.legend(['train', 'test'], loc='upper left')

    return fig, axes
