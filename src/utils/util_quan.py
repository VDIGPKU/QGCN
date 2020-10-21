import numpy as np

def zig_zag(matrix, size=8):
    code = np.zeros([size*size])
    index = -1
    bound = 0
    for i in range(0, 2 * size -1):
        if i < size:
            bound = 0
        else:
            bound = i - size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                code[index] = matrix[j, i-j]
            else:
                code[index] = matrix[i-j, j]
    return code


def zig_zag_reverse(code, size=8):
    matrix = np.zeros([size, size])
    index = -1
    bound = 0
    for i in range(0, 2 * size -1):
        if i < size:
            bound = 0
        else:
            bound = i - size + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                matrix[j, i-j] = code[index]
            else:
                matrix[i-j, j] = code[index]
    return matrix


def getQM(imgBinData):
    # with open(imgPath, "rb") as fin:
    #     imgBinData = fin.read()
    ind1 = imgBinData.find(b'\xff\xdb')
    QM1Bin = imgBinData[ind1+5 : ind1+5+64]
    QM1 = zig_zag_reverse(list(QM1Bin), 8)

    imgBinData = imgBinData[ind1+2:]
    ind2 = imgBinData.find(b'\xff\xdb')
    QM2Bin = imgBinData[ind2+5 : ind2+5+64]
    QM2 = zig_zag_reverse(list(QM2Bin), 8)

    return [QM1.astype(np.uint8), QM2.astype(np.uint8)]
