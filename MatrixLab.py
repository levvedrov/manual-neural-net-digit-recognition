class matrix:
    def __init__(self, data):
        rowlen = len(data[0])
        if len(data)==0:
            raise Exception("The data list is empty")
        for index in range(1,len(data)):
            if len(data[index]) != rowlen:
                   raise Exception(f"line {index+1} is invalid")
            
          
        self.matrix = data
        self.height = len(self.matrix)
        self.width = len(self.matrix[0])  
        
    def __add__(self, other):
        sumMatrix = []
        if self.height != other.height or self.width != other.width:
            raise Exception("invalid matrix for addition")

        try:
            raws1 = [] 
            raws2 = []
            for raw in self.matrix:
                raws1.append(raw)
            for raw in other.matrix:
                raws2.append(raw)     
            for index in range(len(raws1)):
                sumMatrix.append([])
                for ival in range(len(raws1[index])):
                    sumMatrix[index].append(raws1[index][ival]+raws2[index][ival])
            return matrix(sumMatrix)
        except Exception as e:
            print(f"Error: {e}")

    def __sub__(self, other):
        subMatrix = []
        if self.height != other.height or self.width != other.width:
            raise Exception("invalid matrix for subtraction")
        else:
            for index in range(len(self.matrix)):
                if len(self.matrix[index]) != len(other.matrix[index]):
                    raise Exception("Invalid amount of the values")
        try:
            raws1 = [] 
            raws2 = []
            for raw in self.matrix:
                raws1.append(raw)
            for raw in other.matrix:
                raws2.append(raw)     
            for index in range(len(raws1)):
                subMatrix.append([])
                for ival in range(len(raws1[index])):
                    subMatrix[index].append(raws1[index][ival]-raws2[index][ival])
            return matrix(subMatrix)
        except Exception as e:
            print(f"Error: {e}")

    def __matmul__(self, other):
        if self.width != other.height:
            raise Exception("Invalid matrix for multiplication")
        newMatrix = [] 
        for RowIndex in range(len(self.matrix)):
            tmpRow = []
            
            for OtherColumnIndex in range(other.width):
                tmpSum = 0
                for UnitIndex in range(len(self.matrix[RowIndex])):
                        tmpSum += self.matrix[RowIndex][UnitIndex]*other.matrix[UnitIndex][OtherColumnIndex]
                tmpRow.append(tmpSum)
            newMatrix.append(tmpRow)

        return matrix(newMatrix)

    def solveGauss(self):
        
        def __elimination():
            newMatrix = self.matrix
            yend = 0 
            xend = self.width-2
            j=0
            for x in range(0, xend):
                for y in range(self.height-1, yend, -1):

                    
                    tmpRow = []
                    const = self.matrix[y][x]/self.matrix[j][x]
                    for index in range(self.width):
                        subject = self.matrix[y][index]
                        subtraction = self.matrix[j][index] * const
                        tmpRow.append(subject - subtraction)
                    newMatrix[y]=tmpRow
                j+=1 # отступ на верхнюю строчку с нулем
                yend += 1 
                xend-=1
            return newMatrix
                

        def __backSubstitution(matrix):
            roots = []
            for i in range(len(matrix[0])-1):
                roots.append(None)
            
            
            for rowIndex in range(len(matrix)-1, -1, -1):
                unitIndex = 1
                if  rowIndex == self.height-1:
                    unitIndex = self.width-2
                    div = matrix[rowIndex][unitIndex]
                    roots[unitIndex] = matrix[rowIndex][len(matrix[rowIndex])-1]/div

                else:
                    sub = 0
                    for index in range(len(matrix[rowIndex])-1):
                        if roots[index] == None:
                            div = matrix[rowIndex][index]
                        else:
                            sub += matrix[rowIndex][index]*roots[index]
                            unitIndex+=1
    
                    roots[len(roots)-unitIndex] = (matrix[rowIndex][len(matrix[rowIndex])-1]-sub)/div
            return roots

        return __backSubstitution(__elimination())

    def show(self):
        try:
            for raw in self.matrix:
                print(f"|", end=' ')
                for val in raw:
                    print(f"{val} ", end='')
                print(f"|")
            print(f" ")
        except Exception as e:
            print(f"Error: {e}")

    def transpose(self):
        newMatrix = []
        for indexRow in range(self.width):
            row = [] 
            for r in self.matrix:
                row.append(r[indexRow])
            newMatrix.append(row)
        return matrix(newMatrix)

    def __mul__(self, val):
        if isinstance(val, int) or isinstance(val, float):
            newMatrix = []
            for row in self.matrix:
                tmpRow = []
                for unit in row:
                    tmpRow.append(unit*val)
                newMatrix.append(tmpRow)
            return matrix(newMatrix)
        else:
            if self.height!=val.height or self.width !=val.width:
                raise Exception("Matrices must have the same size for element-wise multiplication")
            newMatrix = []
            for irow in range(val.height):
                tmpRow = []
                for iunit in range(val.width):
                    tmpRow.append(self.matrix[irow][iunit]*val.matrix[irow][iunit])
                newMatrix.append(tmpRow)

            return matrix(newMatrix)
