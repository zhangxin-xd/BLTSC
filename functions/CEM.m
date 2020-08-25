function Zcem = CEM(HIM, d)
    [x, y, z] = size(HIM);
    R = zeros(z);
    r = reshape(HIM, x*y, z);
    r = transpose(r);
    R = r*r';
    R = R/(x*y);
    w = (R\d)/(transpose(d)*(R\d));
    for i = 1:x*y
        Z(i) = transpose(w)*r(:,i);
    end
    Zcem = reshape(Z,x,y);
    Zcem = abs(Zcem);
end