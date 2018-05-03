clear all;
u=0;t=1;numax=20;
for iu=1:numax
    uu(iu)=(iu-1)*0.4;
    u=uu(iu);
for i=1:36
    for j=1:36
        a(i,j)=0;
    end
end
for i=1:6
    a(i,i)=0;
end
for i=7:30
    a(i,i)=u;
end
for i=31:36
    a(i,i)=2*u;
end
a(1,26)=t;a(1,13)=t;a(1,7)=t;a(1,21)=-t;a(1,15)=-t;
a(1,29)=t;a(1,23)=t;a(1,12)=t;
a(2,25)=-t;a(2,14)=-t;a(2,8)=-t;a(2,22)=t;
a(2,16)=t;a(2,30)=-t;a(2,24)=-t;a(2,11)=-t;
a(3,25)=t;a(3,21)=t;a(3,15)=t;a(3,11)=t;
a(4,14)=t;a(4,8)=t;a(4,29)=-t;a(4,23)=-t;
a(5,26)=-t;a(5,22)=-t;a(5,16)=-t;a(5,12)=-t;
a(6,13)=-t;a(6,7)=-t;a(6,30)=t;a(6,24)=t;
a(7,28)=t;a(7,6)=-t;a(7,1)=t;a(7,9)=t;a(7,33)=t;a(7,32)=t;
a(8,2)=-t;a(8,27)=t;a(8,4)=t;a(8,10)=t;a(8,33)=-t;a(8,32)=-t;
a(9,30)=t;a(9,15)=-t;a(9,7)=t;a(9,11)=t;
a(10,16)=-t;a(10,29)=t;a(10,8)=t;a(10,12)=t;
a(11,2)=-t;a(11,3)=t;a(11,17)=-t;a(11,32)=t;a(11,31)=t;a(11,9)=t;
a(12,5)=-t;a(12,18)=-t;a(12,1)=t;a(12,32)=-t;a(12,31)=-t;a(12,10)=t;
a(13,1)=t;a(13,6)=-t;a(13,19)=-t;a(13,35)=t;a(13,34)=t;a(13,18)=-t;
a(14,4)=t;a(14,20)=-t;a(14,2)=-t;a(14,35)=-t;a(14,34)=-t;a(14,17)=-t;
a(15,35)=t;a(15,1)=-t;a(15,3)=t;a(15,9)=-t;a(15,17)=t;a(15,31)=t;
a(16,35)=-t;a(16,10)=-t;a(16,5)=-t;a(16,2)=t;a(16,18)=t;a(16,31)=-t;
a(17,14)=-t;a(17,23)=-t;a(17,11)=-t;a(17,15)=t;
a(18,13)=-t;a(18,12)=-t;a(18,24)=-t;a(18,16)=t;
a(19,21)=t;a(19,25)=-t;a(19,13)=-t;a(19,24)=-t;
a(20,22)=t;a(20,14)=-t;a(20,26)=-t;a(20,23)=-t;
a(21,36)=t;a(21,19)=t;a(21,3)=t;a(21,27)=-t;a(21,1)=-t;a(21,32)=t;
a(22,36)=-t;a(22,20)=t;a(22,2)=t;a(22,5)=-t;a(22,28)=-t;a(22,32)=-t;
a(23,20)=-t;a(23,34)=t;a(23,32)=t;a(23,17)=-t;a(23,4)=-t;a(23,1)=t;
a(24,19)=-t;a(24,34)=-t;a(24,32)=-t;a(24,2)=-t;a(24,18)=-t;a(24,6)=t;
a(25,27)=t;a(25,36)=t;a(25,35)=t;a(25,19)=-t;a(25,3)=t;a(25,2)=-t;
a(26,28)=t;a(26,36)=-t;a(26,35)=-t;a(26,1)=t;a(26,20)=-t;a(26,5)=-t;
a(27,25)=t;a(27,29)=t;a(27,21)=-t;a(27,8)=t;
a(28,26)=t;a(28,30)=t;a(28,7)=t;a(28,22)=-t;
a(29,35)=t;a(29,33)=t;a(29,27)=t;a(29,1)=t;a(29,4)=-t;a(29,10)=t;
a(30,35)=-t;a(30,33)=-t;a(30,28)=t;a(30,6)=t;a(30,9)=t;a(30,2)=-t;
a(31,16)=-t;a(31,15)=t;a(31,12)=-t;a(31,11)=t;
a(32,22)=-t;a(32,24)=-t;a(32,21)=t;a(32,23)=t;
a(32,11)=t;a(32,8)=-t;a(32,12)=-t;a(32,7)=t;
a(33,30)=-t;a(33,29)=t;a(33,7)=t;a(33,8)=-t;
a(34,23)=t;a(34,24)=-t;a(34,14)=-t;a(34,13)=t;
a(35,29)=t;a(35,26)=-t;a(35,30)=-t;a(35,25)=t;
a(35,13)=t;a(35,15)=t;a(35,14)=-t;a(35,16)=-t;
a(36,25)=t;a(36,26)=-t;a(36,21)=t;a(36,22)=-t;
[ev,ew]=eig(a);
for ien=1:36
    eigen(ien,iu)=ew(ien,ien);
end
end
for ien=1:36
plot(uu,eigen(ien,:))
hold on
end
