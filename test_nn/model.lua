-- Rnn Model

local Rnn, parent = torch.class('Rnn', 'nn.Module')

function Rnn:__init(config)
  parent.__init(self)
  self.in_dim = config.in_dim
  self.mem_dim = config.mem_dim or 150
  self.batch_size = config.batch_size
  self.mem_zeros = torch.zeros(self.batch_size, self.mem_dim)
  self.gpu = config.gpu
  self.max_sequence_length = config.max_sequence_length
  self.num_classes = 2
  self.children = {}

  -- composition module
  self.composer = self:new_composer()
  self.composers = {}
  print('creating composers')
  self:create_composers()
  print(#self.composers)
  print('done')

  -- tensor tables
  self.one_dim_tensors = {}
  self.three_dim_tensors = {}

  -- -- output module
  self.output_module = self:create_output_module()

  -- criterion module
  self.criterion = nn.ClassNLLCriterion()

  if self.gpu then
    self.mem_zeros = self.mem_zeros:cuda()
    self.criterion:cuda()
  end
end

function Rnn:create_composers()
  for i = 1, self.max_sequence_length do
    self.composers[i] = self:new_composer()
  end
end

-- from last hidden state to output.
function Rnn:create_output_module()
  local output_module = nn.Sequential()
  if self.dropout then
    sentiment_module:add(nn.Dropout())
  end
  output_module
    :add(nn.Linear(self.mem_dim, self.num_classes))
    :add(nn.LogSoftMax())
  if self.gpu then
    output_module:cuda()
  end
  return output_module
end

function Rnn:new_composer()
  local input = nn.Identity()()
  local child_c = nn.Identity()()
  local child_h = nn.Identity()()
  local child_h_sum = nn.Sum(2)(child_h)
  local m
  if self.gpu then
    m = cudnn
  else
    m = nn
  end

  local Wx = nn.Linear(self.in_dim, 3*self.mem_dim)(input)
  local Wh_o_i_u = nn.Linear(self.mem_dim, 3*self.mem_dim)(child_h_sum)

  local Wx_o_i_u = nn.Narrow(2, 1, 3*self.mem_dim)(Wx)
  local Wx_f = nn.Narrow(2, 3*self.mem_dim+1, self.mem_dim)(Wx)

  local add_o_i_u = nn.CAddTable(){Wx_o_i_u, Wh_o_i_u}

  local o_i = m.Sigmoid()(nn.Narrow(2, 1, 2*self.mem_dim)(add_o_i_u))
  local u = m.Tanh()(nn.Narrow(2, self.mem_dim*2+1, self.mem_dim)(add_o_i_u))

  local o = nn.Narrow(2, 1, self.mem_dim)(o_i)
  local i = nn.Narrow(2, self.mem_dim+1, self.mem_dim)(o_i)

  local f = m.Sigmoid()(
    treelstm.CRowAddTable(){
      m.TemporalConvolution(self.mem_dim, self.mem_dim, 1)(child_h),
      Wx_f
  })
  
  local c = nn.CAddTable(){
    nn.CMulTable(){i, u},
    nn.Sum(2)(nn.CMulTable(){f, child_c})
  }

  local h = nn.CMulTable(){o, m.Tanh()(c)}

  local composer = nn.gModule({input, child_c, child_h}, {c, h})
  if self.gpu then
    composer:cuda()
  end
  if self.composer ~= nil then
    share_params(composer, self.composer)
  end
  return composer
end

function Rnn:forward(tree, inputs)
  loss = 0
  for i = 1, tree.num_children do
    _, child_loss = self:forward(tree.children[i], inputs)
    loss = loss + child_loss
  end
  
  local i = tree.idx
  local child_c, child_h = self:get_child_states(tree, self.batch_size)
  self.children[i] = {child_c, child_h}
  tree.state = self.composers[i]:forward({inputs[i], child_c, child_h})

  if tree.root then
    tree.output = self.output_module:forward(tree.state[2])
    if self.train then
      loss = loss + self.criterion:forward(tree.output, tree.gold_labels)
    end
  end

  return tree.state, loss
end

function Rnn:backward(tree, inputs, grad, batch_size)
  local input_size = inputs:size()
  local grad_inputs
    if self.gpu then
      grad_inputs = torch.CudaTensor(inputs:size())
    else
      grad_inputs = torch.Tensor(inputs:size())
    end
  self:_backward(tree, inputs, grad, grad_inputs, batch_size)
  return grad_inputs
end

function Rnn:_backward(tree, inputs, grad, grad_inputs, batch_size)
  local output_grad = self.mem_zeros
  local idx = tree.idx
  if tree.output ~= nil and tree.gold_labels ~= nil then
    local criterion_grad = self.criterion:backward(tree.output, tree.gold_labels)
    output_grad = self.output_module:backward(
      tree.state[2], criterion_grad)
  end
  local children = self.children[idx]
  local composer_grad = self.composers[idx]:backward(
    {inputs[idx], children[1], children[2]},
    {grad[1], grad[2]:add(output_grad)})
  grad_inputs[idx] = composer_grad[1]
  local child_c_grads, child_h_grads = composer_grad[2], composer_grad[3]
  for i = 1, tree.num_children do
    self:_backward(tree.children[i], inputs, {child_c_grads[{{},{i}}]:squeeze(), child_h_grads[{{},{i}}]:squeeze()}, grad_inputs, batch_size)
  end
end

function Rnn:get_child_states(tree, batch_size)
  local child_c, child_h

  if tree.num_children == 0 then
    if self.gpu then
      child_c = torch.zeros(batch_size, 1, self.mem_dim):cuda()
      child_h = torch.zeros(batch_size, 1, self.mem_dim):cuda()
    else
      child_c = torch.zeros(batch_size, 1, self.mem_dim)
      child_h = torch.zeros(batch_size, 1, self.mem_dim)
    end
  else  
    if self.gpu then
      child_c = torch.CudaTensor(batch_size, tree.num_children, self.mem_dim)
      child_h = torch.CudaTensor(batch_size, tree.num_children, self.mem_dim)
    else
      child_c = torch.Tensor(batch_size, tree.num_children, self.mem_dim)
      child_h = torch.Tensor(batch_size, tree.num_children, self.mem_dim)
    end
  end

  for i = 1, tree.num_children do
    child_c[{{}, {i}}], child_h[{{}, {i}}] = unpack(tree.children[i].state)
  end
  return child_c, child_h
end

function Rnn:parameters()
  local params, grad_params = {}, {}
  local cp, cg = self.composer:parameters()
  tablex.insertvalues(params, cp)
  tablex.insertvalues(grad_params, cg)

  local op, og = self.output_module:parameters()
  tablex.insertvalues(params, op)
  tablex.insertvalues(grad_params, og)
  return params, grad_params
end
 
function Rnn:training()
  self.train = true
end

function Rnn:evaluating()
  self.train = false
end
