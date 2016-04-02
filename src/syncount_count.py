def get_model(opt):

    inp_size = opt['inp_size']
    inp_dim = opt['inp_dim']
    num_proposals = opt['num_proposals']
    # Convolution feature map of the image.
    inp = tf.placeholder('float', [None, inp_size, inp_size, inp_dim])

    # Instance segmentation proposal mask
    # Fixed length of 20 masks.
    segm_masks = tf.placeholder(
        'float', [None, num_proposals, inp_size, inp_size])

    # 1st convolution layer
    w_conv1 = weight_variable([5, 5, 3, 16])
    b_conv1 = weight_variable([16])
    h_conv1 = tf.nn.relu(conv2d(inp, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    pass
