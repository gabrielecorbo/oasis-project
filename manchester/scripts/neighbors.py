import numpy as np

def neighbors(rows,colums,index):
    mat=np.zeros((rows,colums))
    k=0
    for j in range(colums):
        for i in range(rows)[::-1]:
            mat[i,j]=k
            k+=1
    print(mat)
    i_inx=rows-index%rows-1
    j_inx=int(index/rows)
    print(i_inx,j_inx)
    v=[]
    if (i_inx-1)>=0:
        v.append(mat[i_inx-1,j_inx])
    if (i_inx+1)<rows:
        v.append(mat[i_inx+1,j_inx])
    if (j_inx-1)>=0:
        v.append(mat[i_inx,j_inx-1])
    if (j_inx+1)<colums:
        v.append(mat[i_inx,j_inx+1])
    if (rows-i_inx)%2==0:   #seconda quarta sesta ecc riga dal basso
        if (i_inx+1)<rows and (j_inx+1)<colums:
            v.append(mat[i_inx+1,j_inx+1])
        if (i_inx-1)>=0 and (j_inx+1)<colums:
            v.append(mat[i_inx-1,j_inx+1])
    else:
        if (i_inx+1)<rows and (j_inx-1)>=0:
            v.append(mat[i_inx+1,j_inx-1])
        if (i_inx-1)>=0 and (j_inx-1)>=0:
            v.append(mat[i_inx-1,j_inx-1])
    print(v)

    ##### second level
    v2=[]
    if (i_inx-2)>=0:
        v2.append(mat[i_inx-2,j_inx])
    if (i_inx+2)<rows:
        v2.append(mat[i_inx+2,j_inx])
    if (j_inx-2)>=0:
        v2.append(mat[i_inx,j_inx-2])
    if (j_inx+2)<colums:
        v2.append(mat[i_inx,j_inx+2])
    if (i_inx+2)<rows and (j_inx+1)<colums:
        v2.append(mat[i_inx+2,j_inx+1])
    if (i_inx+2)<rows and (j_inx-1)>=0:
        v2.append(mat[i_inx+2,j_inx-1])
    if (i_inx-2)>=0 and (j_inx+1)<colums:
        v2.append(mat[i_inx-2,j_inx+1])
    if (i_inx-2)>=0 and (j_inx-1)>=0:
        v2.append(mat[i_inx-2,j_inx-1])
    if (rows-i_inx)%2==0:   #seconda quarta sesta ecc riga dal basso
        if (i_inx+1)<rows and (j_inx-1)>=0:
            v2.append(mat[i_inx+1,j_inx-1])
        if (i_inx-1)>=0 and (j_inx-1)>=0:
            v2.append(mat[i_inx-1,j_inx-1])
        if (i_inx+1)<rows and (j_inx+2)<colums:
            v2.append(mat[i_inx+1,j_inx+2])
        if (i_inx-1)>=0 and (j_inx+2)<colums:
            v2.append(mat[i_inx-1,j_inx+2])
    else:
        if (i_inx+1)<rows and (j_inx-2)>=0:
            v2.append(mat[i_inx+1,j_inx-2])
        if (i_inx-1)>=0 and (j_inx-2)>=0:
            v2.append(mat[i_inx-1,j_inx-2])
        if (i_inx+1)<rows and (j_inx+1)<colums:
            v2.append(mat[i_inx+1,j_inx+1])
        if (i_inx-1)>=0 and (j_inx+1)<colums:
            v2.append(mat[i_inx-1,j_inx+1])
    print(v2)
    return v,v2


neighbors(6,6,0)
    
     