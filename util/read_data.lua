--[[

  Functions for loading data from disk.

--]] 
--need next batch function
function treelstm.read_web_dataset(dir)
  local dataset = {}
  input_size = 2
  local num_examples = 500
  encoding_file_name = 'encodings.txt'
  local tree_file_name = 'trees.txt'
  local label_file_name = 'labels.txt'
  if treelstm.files[encoding_file_name] == nil then
    treelstm.open_files(dir, encoding_file_name, tree_file_name, label_file_name)
  end

  print('processing trees')
  local children = treelstm.read_3D_data(dir, tree_file_name, num_examples)

  dataset.size = #encodings
  dataset.encodings = encodings
  dataset.trees = treelstm.read_trees(children, labels)
  dataset.labels = torch.Tensor(dataset.size) 

  for i = 1, dataset.size do
    dataset.labels[i] = dataset.trees[i].gold_label
  end
  
  if dataset.size ~= 0 then
    print('dataset size: ' .. dataset.size)
    dataset.input_dim = dataset.encodings[1][1]:size(1)
  else
    treelstm.files = {}
  end
  return dataset
end

function mock_dataset(children)
  local encodings = {}
  local trees = {}
  local labels = {}

  for i=1, #children do --iterate thru sites
    local tree = children[i]
    encodings[i] = torch.Tensor(#tree, input_size)
    local target = math.random(#tree)
    for j=1, #tree do -- for each node in site
      local target_node = torch.Tensor(input_size):zero()
      if j == target then 
        if math.random() > .5 then
          target_node[1] = 1
        end
        if math.random() > .5 then
          target_node[2] = 1
        end
        if target_node[1] ==1 and target_node[2] == 1 then
          labels[i] = 2
        else
          labels[i] = 1
        end
      end
      encodings[i][j] = target_node
    end
  end
  return {encodings, labels}
end

function verify_data(encodings, trees, labels)
  local counter = 1
  local num_nodes = 0
  if #encodings ~= #trees or #encodings ~= #labels then
    print('ERROR number of examples dont match')
    print('number of encodings: ' .. #encodings)
    print('number of trees: ' .. #trees)
    print('number of labels: ' .. #labels)
  end

  for i=1, #encodings do
    local tree = trees[i]
    local label = labels[i]
    local num_nodes_temp = encodings[i]:size(1)
    if num_nodes == num_nodes_temp then
      counter = counter + 1
    else
      counter = 1
    end
    num_nodes = num_nodes_temp

    local encoding = encodings[i] -- one site
    local label = labels[i]
    for i=1, (#encoding)[1] do 
      local node = encoding[i] -- one node
      local fontOnOff = node[575]
      if fontOnOff == 1 then
        fontOnOff = 2 -- class 2: font is set
      else
        fontOnOff = 1 -- class 1: font is not set
      end
      if node[1] == 1 then
        if label == fontOnOff then
          -- print('label: ' .. label .. ' matches encoding: ' .. fontOnOff)
        else
          print('ERROR: Label/Encoding mismatch: ' .. label .. ' / ' .. fontOnOff)
        end
      end
    end

    for c=1, #tree do
      local child_index = tonumber(tree[c][1])
      if child_index ~= nil and child_index > num_nodes then
      end
    end
  end
end



function treelstm.open_files(dir, ...)
  local file_names = {...}
  for i, file_name in ipairs(file_names) do
    local path = dir .. file_name
    print('opening file found at: ' .. path)
    local file = io.open(path)
    if file == nil then
      print('ERROR: file not found')
      os.exit()
    end
    treelstm.files[file_name] = file
  end
end

function treelstm.read_labels(dir, file_name, num_examples)
  file = treelstm.files[file_name]
  local labels = {}
  for c=1, num_examples do
    local line = file:read()
    if line == nil then break end
    labels[c] = tonumber(line)
  end
  return labels
end

function treelstm.read_3D_data(dir, file_name, num_examples)
  local file = treelstm.files[file_name]
  local index = 0
  local first_dim = {}
  local max_sequence_length = 0

  for c=1, num_examples do
    local line = file:read()
    if line == nil then break end
    local second_dim = stringx.split(line, '\t') -- table of nodes within one site
    local encoding_length_counter = 0

    if #second_dim > max_sequence_length then
      max_sequence_length = #second_dim
    end

    for i, n in ipairs(second_dim) do -- iterate through one site's nodes
      local third_dim = stringx.split(n) -- table of individual node's encoding
      second_dim[i] = third_dim
    end
    if file_name == encoding_file_name then
      for i, n in ipairs(second_dim) do
        local size = #n
        if size ~= 595 then
          print('ERROR: encoding is of length: ' .. size .. ' . Should be 595')
        end
      end
      second_dim = torch.Tensor(second_dim)
    end
    index = index + 1
    first_dim[index] = second_dim
  end

  -- first_dim.max_sequence_length = max_sequence_length
  return first_dim
end

function treelstm.read_trees(all_children, all_labels)
  local trees = {}
  for i, site_children in ipairs(all_children) do -- iterate through training examples (one site)
    trees[i] = treelstm.read_tree(site_children, all_labels[i], 0)
  end
  return trees
end

--function starts at root of one site's tree and recursively visits children
function treelstm.read_tree(site_children, site_label, idx)
  local node_children = site_children[idx+1]
  local tree = treelstm.Tree()
  tree.idx = idx+1
  if tree.idx == 1 then
    tree.gold_label = torch.Tensor(1)
    tree.gold_label[1] = site_label
  else 
    tree.gold_label = nil
  end

  if node_children ~= nil then
    for i, child_idx in ipairs(node_children) do
      local child_tree = treelstm.read_tree(site_children, site_label, child_idx)
      tree:add_child(child_tree)
    end
  end
  return tree
end
