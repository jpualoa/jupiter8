import os

def log_training(filename, epoch, d_loss, g_loss, tstamp):
    msg = "epoch:%s,d_loss:%f,d_acc:%.2f,g_loss:%f,time:%.2f\n" % (
        epoch, d_loss[0], 100*d_loss[1], g_loss, tstamp)
    if os.path.exists(filename): method = 'a'
    else: method = 'w'
    with open(filename, method) as f:
        f.write(msg)
