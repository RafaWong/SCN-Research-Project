function a = formatConvert(x)
a = '';
for i = 1:length(x)
    a = [a,num2str(x(i)),','];
end
a(end)='';