require('..')
include('model.lua')

local batch_size = 25
local mem_dim = 1024
local in_dim = 2
local gpu = treelstm.use_gpu
local num_epochs = 100
local learning_rate = .02
local zeros = torch.zeros(batch_size, mem_dim)
local optim_state = { learningRate = learning_rate}
local data_dir = 'data/web/'
local reg = 1e-4 
local input_tensors = {}
-- nngraph.setDebug(true)

if gpu then 
  require('cutorch')
  require('cunn')
  require('cudnn')
  cudnn.fastest = true
  memory_usage = cutorch.getMemoryUsage()
end

-- load training dataset
local train_dir = data_dir .. 'train/'

-- create Rnn module
local config = {
  in_dim = in_dim,
  mem_dim = mem_dim,
  gpu = gpu,
  batch_size = batch_size,
  max_sequence_length = 500
}
local rnn = Rnn(config)
local params, grad_params = rnn:getParameters()

function train()
  for i = 1, num_epochs do
    local path = i .. '.th'
    print('saving model to ' .. path)
    torch.save(path, {
      parameters = params
    })
    _train(i)
  end
end

-- train dataset
function _train(epoch)
  rnn:training()
  while true do
    local dataset = treelstm.read_web_dataset(train_dir)
    if dataset.size == 0 then break end
    for i = 1, dataset.size, batch_size do
      local calculated_batch_size = math.min(i + batch_size - 1, dataset.size) - i + 1
      if calculated_batch_size ~= batch_size then break end
      local sequence_length = dataset.encodings[i]:size(1)

      dataset.trees[i] = dataset.trees[i] or {}
      if gpu then
        local temp = cutorch.getMemoryUsage()
        local result = memory_usage - temp
        memory_usage = temp
      end

      local feval = function(x)
        grad_params:zero()
        local tree = dataset.trees[i]
        tree.root = true
        local inputs 
        local labels 
          
        if gpu then
          inputs = torch.CudaTensor(sequence_length, batch_size, in_dim)
          labels = torch.CudaTensor(batch_size)
        else
          inputs = torch.Tensor(sequence_length, batch_size, in_dim)
          labels = torch.Tensor(batch_size)
        end
          
        for k = 1, sequence_length do
          for j = 1, batch_size do
            local idx = i+j-1--indices[i + j - 1]
            local encodings = dataset.encodings[idx]
            local node = encodings[k]
            inputs[k][j] = node
          end
        end

        for k = 1, batch_size do
          local idx = i + k - 1--indices[i + k - 1]
          labels[k] = dataset.trees[idx].gold_label
        end
        tree.gold_labels = labels
        local _, loss = rnn:forward(tree, inputs)
        loss = loss / batch_size

        local input_grad = rnn:backward(tree, inputs, {zeros, zeros}, batch_size)
        loss = loss / batch_size
        grad_params:div(batch_size)

        loss = loss + 0.5 * reg * params:norm() ^ 2
        grad_params:add(reg, params)
        return loss, grad_params
      end
      optim.adagrad(feval, params, optim_state)
    end
  end
end

function accuracy(pred, gold)
  local result = torch.abs(pred:csub(gold)):sum()
  return result
end

-- prediction helper function
function _predict(tree, sent)
  rnn:evaluate()
  local prediction
  local inputs = rnn:forward(tree, inputs)
  return tree.loss
end

-- predict dev set
function predict(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    predictions[i] = _predict(dataset.trees[i], dataset.sents[i])
  end
  return predictions
end

if gpu then
  zeros = zeros:cuda()
end
train()