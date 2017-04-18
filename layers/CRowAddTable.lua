--[[
  
  Add a vector to every row of a matrix.

  Input: { [n x m], [m] }

  Output: [n x m]

--]]

local CRowAddTable, parent = torch.class('treelstm.CRowAddTable', 'nn.Module')

function CRowAddTable:__init()
  parent.__init(self)
  self.gradInput = {}
  self.outputTensors = {}
end

function CRowAddTable:updateOutput(input)
  self.output:resizeAs(input[1]):copy(input[1])

  if input[1]:nDimension() == 2 then
    for i = 1, self.output:size(1) do
      self.output[i]:add(input[2])
    end
  else
    for i = 1, self.output:size(1) do
      for j = 1, self.output:size(2) do
        self.output[i][j]:add(input[2][i])
      end
    end
  end
  return self.output
end

function CRowAddTable:get_tensor(input_size, existing_tensor)
  local num_dimensions = input_size:size()
  local input_size1 = input_size[1]
  local input_size2 = input_size[2]
  local input_size3
  if existing_tensor == nil then
    print('creating new CRowAddTable tensor')
    if num_dimensions == 2 then
      return get_tensor(input_size1, input_size2)
    elseif num_dimensions == 3 then
      input_size3 = input_size[3]
      return get_tensor(input_size1, input_size2, input_size3)
    end
  else
    local existing_tensor_size = existing_tensor:size()
    local existing_tensor_size1 = existing_tensor_size[1]
    local existing_tensor_size2 = existing_tensor_size[2]
    if num_dimensions == 3 then
      local input_size3 = input_size[3]
      local existing_tensor_size3 = existing_tensor_size[3]
      if existing_tensor_size1 ~= input_size1 or existing_tensor_size2 ~= input_size2 or
         existing_tensor_size3 ~= input_size3 then
        print('creating new CRowAddTable tensor')
        return get_tensor(input_size1, input_size2, input_size3)
      end
    else
      if existing_tensor_size1 ~= input_size1 or existing_tensor_size2 ~= input_size2 then
        print('creating new CRowAddTable tensor')
        return get_tensor(input_size1, input_size2)
      end
    end
  end
  return existing_tensor
end

function equal_size(size1, size2)
  if size1:size() ~= size2:size() then
    return false
  else
    local nDim = size1:size()
    if nDim == 1 and size1[1] == size2[1] then
      return true
    elseif nDim == 2 and size1[1] == size2[1] and size1[2] == size2[2] then
      return true
    elseif nDim == 3 and size1[1] == size2[1] and size1[2] == size2[2] and size1[3] == size2[3] then
      return true
    end
  end
  return false
end



function CRowAddTable:updateGradInput(input, gradOutput)
  local input1_size = input[1]:size()
  local input2_size = input[2]:size()

  if self.gradInput[1] == nil then
    if treelstm.use_gpu then
      self.gradInput[1] = torch.CudaTensor(input1_size)
    else
      self.gradInput[1] = torch.Tensor(input1_size)
    end
  elseif equal_size(input1_size, self.gradInput[1]:size()) ~= true then
    if treelstm.use_gpu then
      self.gradInput[1] = torch.CudaTensor(input1_size)
    else
      self.gradInput[1] = torch.Tensor(input1_size)
    end
  end

  if self.gradInput[2] == nil then
    if treelstm.use_gpu then
      self.gradInput[2] = torch.CudaTensor(input2_size)
    else
      self.gradInput[2] = torch.Tensor(input2_size)
    end
  elseif equal_size(input2_size, self.gradInput[2]:size()) ~= true then
    if treelstm.use_gpu then
      self.gradInput[2] = torch.CudaTensor(input2_size)
    else
      self.gradInput[2] = torch.Tensor(input2_size)
    end
  end

   self.gradInput[1]:copy(gradOutput)
   if gradOutput:nDimension() == 2 then
     for i = 1, gradOutput:size(1) do
        self.gradInput[2]:add(gradOutput[i])
     end
   else 
     for i = 1, gradOutput:size(1) do
       for j = 1, gradOutput:size(2) do
         self.gradInput[2][i]:add(gradOutput[i][j])
       end
     end
   end

   return self.gradInput
end
