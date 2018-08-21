%% Input
dataset_path = '<input_dataset>/features/fullFrame-210x260px';

%% Ouput
output_path =  '<output_dataset>/features/fullFrame-227x227px';

%%
% reshape_size = [256 256]
reshape_size = [227 227];

data_partitions = dir(dataset_path);
data_partitions=data_partitions(~ismember({data_partitions.name},{'.','..'}));

for i = 1:numel(data_partitions)
    
    curr_partition_path = [dataset_path '/', data_partitions(i).name];
    curr_sequences = dir(curr_partition_path);
    curr_sequences = curr_sequences(~ismember({curr_sequences.name},{'.','..'}));
    curr_sequences(~[curr_sequences(:).isdir]) = [];
    
    for j = 1:numel(curr_sequences)
        
        curr_sequence_path = [curr_partition_path, '/', curr_sequences(j).name '/'];
        curr_frames = dir(curr_sequence_path);  
        curr_frames = curr_frames(~ismember({curr_frames.name},{'.','..'}));
        curr_output_path = [output_path, '/', data_partitions(i).name, '/', curr_sequences(j).name];
        if ~(exist(curr_output_path,'dir'))
            mkdir(curr_output_path)
            disp(curr_sequence_path);
            for f = 1:numel(curr_frames)
                curr_frame = [curr_sequence_path, '/',curr_frames(f).name];
                curr_output_frame = [curr_output_path, '/', curr_frames(f).name];
                imwrite(imresize(imread(curr_frame),reshape_size), curr_output_frame)% ,'Method', 'lanczos3');
            end
        end
    end
end