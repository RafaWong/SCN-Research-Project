function data_trim = cellset2trim(dataset,trim_len)

[cellNum,timeline] = size(dataset);
data_trim = cell(size(dataset));
for ii = 1:cellNum
    for jj = 1:timeline
        temp = dataset{ii,jj};
        if isempty(temp) == 0
            data_trim{ii,jj} = temp(1:trim_len,:);
        end
    end
end

end