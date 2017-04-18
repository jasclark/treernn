require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

treelstm = {}

include('util/read_data.lua')
include('util/Tree.lua')
include('layers/CRowAddTable.lua')

printf = utils.printf
torch.setheaptracking(true)

-- global paths (modify if desired)
treelstm.data_dir        = 'data'
treelstm.models_dir      = 'trained_models'
treelstm.predictions_dir = 'predictions'

treelstm.use_gpu = false
treelstm.files = {}

-- share module parameters
function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

local one_dim_tensors = {}
local two_dim_tensors = {}
local three_dim_tensors = {}

function get_tensor(...)
  if treelstm.use_gpu then
    torch.setdefaulttensortype('torch.CudaTensor')
  end
  local args = {...}
  local num_dim = #args
  local size1 = args[1]
  local tensor

  --1-D tensor
  if num_dim == 1 then
    tensor = one_dim_tensors[size1]
    if tensor == nil then
      print('creating new 1D tensor of size ' .. size1)
      tensor = torch.Tensor(size1)
      one_dim_tensors[size1] = tensor
    end
 
  -- 2D tensor
  elseif num_dim == 2 then
    local size2 = args[2]
    local dim1_tensors = get_tensors(size1, two_dim_tensors)
    tensor = dim1_tensors[size2]
    if tensor == nil then
      print('creating new 2D tensor of size: ' .. size1 .. ' x ' .. size2)
      tensor = torch.Tensor(size1, size2)
      dim1_tensors[size2] = tensor
    end

  -- 3D tensor
  elseif num_dim == 3 then
    local size2 = args[2]
    local size3 = args[3]
    local dim1_tensors = get_tensors(size1, three_dim_tensors)
    local dim2_tensors = get_tensors(size2, dim1_tensors)
    tensor = dim2_tensors[size3]
    if tensor == nil then
      print('creating new 3d tensor of size: ' .. size1 .. ' x ' .. size2 .. ' x ' .. size3)
      tensor = torch.Tensor(size1, size2, size3)
      dim2_tensors[size3] = tensor
    end
  end
  torch.setdefaulttensortype('torch.DoubleTensor')
  return tensor
end

function get_tensors(size, tensors)
  if tensors[size] == nil then
    tensors[size] = {}
  end
  return tensors[size]
end