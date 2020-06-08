import torch
from six.moves import xrange

# Virtual adversarial training parameters
num_power_iteration = 1
small_constant_for_finite_diff = 1e-1


def adversarial_loss(embedded, segment_ids, input_mask, document_mask, label_ids, loss, loss_fn, perturb_norm_length):
    """Adds gradient to embedding and recomputes classification loss."""
    grad, = torch.autograd.grad(
      loss,
      embedded,
      retain_graph=True)
    grad.detach_()
    perturb = _scale_l2(grad, perturb_norm_length)
    return loss_fn(token_type_ids=segment_ids, attention_mask=input_mask,
                   document_mask=document_mask, labels=label_ids, input_embeddings=embedded + perturb)


def virtual_adversarial_loss(logits, embedded, segment_ids, input_mask, document_mask,
                             num_classes, logits_from_embedding_fn, perturb_norm_length):
    """Virtual adversarial loss.
    Computes virtual adversarial perturbation by finite difference method and
    power iteration, adds it to the embedding, and computes the KL divergence
    between the new logits and the original logits.
    Args:
    logits: 3-D float Tensor, [batch_size, num_timesteps, m], where m=1 if
      num_classes=2, otherwise m=num_classes.
    embedded: 3-D float Tensor, [batch_size, num_timesteps, embedding_dim].
    inputs: VatxtInput.
    logits_from_embedding_fn: callable that takes embeddings and returns
      classifier logits.
    Returns:
    kl: float scalar.
    """
    # Stop gradient of logits. See https://arxiv.org/abs/1507.00677 for details.
    # logits = tf.stop_gradient(logits)

    # Only care about the KL divergence on the final timestep.
    # weights = inputs.eos_weights
    # assert weights is not None
    # if FLAGS.single_label:
    # indices = tf.stack([tf.range(FLAGS.batch_size), inputs.length - 1], 1)
    # weights = tf.expand_dims(tf.gather_nd(inputs.eos_weights, indices), 1)

    # Initialize perturbation with random noise.
    # shape(embedded) = (batch_size, num_timesteps, embedding_dim)
    d = torch.autograd.Variable(torch.empty(embedded.size()).normal_(), requires_grad=True).cuda()

    # Perform finite difference method and power iteration.
    # See Eq.(8) in the paper http://arxiv.org/pdf/1507.00677.pdf,
    # Adding small noise to input and taking gradient with respect to the noise
    # corresponds to 1 power iteration.
    for _ in xrange(num_power_iteration):
        d = _scale_l2(
            _mask_by_mask(d, input_mask), small_constant_for_finite_diff)

    _, d_logits, _ = logits_from_embedding_fn(token_type_ids=segment_ids, attention_mask=input_mask,
                                            document_mask=document_mask, input_embeddings=embedded + d)
    kl = _kl_divergence_with_logits(logits, d_logits, num_classes)
    perturb, = torch.autograd.grad(
        kl,
        d)
    perturb.detach_()

    perturb = _scale_l2(perturb, perturb_norm_length)
    _, vadv_logits, _ = logits_from_embedding_fn(token_type_ids=segment_ids, attention_mask=input_mask,
                                        document_mask=document_mask, input_embeddings=embedded + perturb)
    return _kl_divergence_with_logits(logits, vadv_logits, num_classes)


def _scale_l2(x, norm_length):
  # shape(x) = (batch, num_timesteps, d)
  # Divide x by max(abs(x)) for a numerically stable L2 norm.
  # 2norm(x) = a * 2norm(x/a)
  # Scale over the full sequence, dims (1, 2)
  alpha = torch.max(torch.max(torch.abs(x), dim=1, keepdim=True)[0], dim=2, keepdim=True)[0] + 1e-12
  l2_norm = alpha * torch.sqrt(
      torch.sum(torch.pow(x / alpha, 2), dim=(1, 2), keepdim=True) + 1e-6)
  x_unit = x / l2_norm
  return norm_length * x_unit


def _mask_by_mask(t, mask):
  """Mask t, 3-D [batch, time, dim], by Mask, 2-D [batch, time]."""

  return t * torch.unsqueeze(mask, dim=2).expand(t.size()).float()


def _kl_divergence_with_logits(q_logits, p_logits, num_classes):
    """Returns weighted KL divergence between distributions q and p.
    Args:
        q_logits: logits for 1st argument of KL divergence shape
                  [batch_size, num_timesteps, num_classes] if num_classes > 2, and
                  [batch_size, num_timesteps] if num_classes == 2.
        p_logits: logits for 2nd argument of KL divergence with same shape q_logits.
        weights: 1-D float tensor with shape [batch_size, num_timesteps].
                 Elements should be 1.0 only on end of sequences
    Returns:
        KL: float scalar.
    """
    # For logistic regression
    if num_classes == 2:
        # q = tf.nn.sigmoid(q_logits)
        # kl = (-tf.nn.sigmoid_cross_entropy_with_logits(logits=q_logits, labels=q) +
        #       tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logits, labels=q))
        # kl = tf.squeeze(kl, 2)
        raise NotImplementedError

    # For softmax regression
    else:
        q = torch.nn.functional.softmax(q_logits, -1)
        kl = torch.sum(
            q * (torch.nn.functional.log_softmax(q_logits, -1) - torch.nn.functional.log_softmax(p_logits, -1)), -1)

    # num_labels = tf.reduce_sum(weights)
    # num_labels = tf.where(tf.equal(num_labels, 0.), 1., num_labels)

    # kl.get_shape().assert_has_rank(2)
    assert len(kl.size()) == 2
    # weights.get_shape().assert_has_rank(2)

    loss = torch.mean(kl)
    return loss