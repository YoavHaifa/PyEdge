# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 00:01:27 2021

@author: yoavb
"""

def atoi(str):
    resultant = 0
    for i in range(len(str)):
        dig = ord(str[i]) - ord('0')
        if dig < 0 or dig > 9:
            break
        resultant = resultant * 10 + dig        #It is ASCII substraction 
    return resultant

def GetValue(sfName, sKey, default = 0):
    i = sfName.find(sKey)
    if i < 0:
        return default
    i += len(sKey)
    return atoi(sfName[i:len(sfName)])

def GetValue2(sfName, sKey, sKey2, default = 0):
    i = sfName.find(sKey)
    if i >= 0:
        i += len(sKey)
        sRest = sfName[i:len(sfName)]
        if sRest[0:1].isdigit():
            return atoi(sfName[i:len(sfName)])

    return GetValue(sfName, sKey2, default)

def SetValue(sfName, sKey, value):
    iKey = sfName.find(sKey)
    if iKey < 0:
        # Add new field
        iDot = sfName.find('.')
        if iDot < 0:
            return sfName + sKey + str(value)
        return sfName[0:iDot] + sKey + str(value) + sfName[iDot:]
    
    sRest = sfName[iKey+1:]
    iEnd = sRest.find('_')
    if iEnd < 0:
        iEnd = sRest.find('.')
    #print(f'sRest {sRest}, iEnd {iEnd}')
    if iEnd > 0:
        sRest = sRest[iEnd:]
        return sfName[0:iKey] + sKey + str(value) + sRest
    return sfName[0:iKey] + sKey + str(value)

def RemoveValue(sfName, sKey):
    iKey = sfName.find(sKey)
    if iKey < 0:
        return sfName
    sRest = sfName[iKey+1:]
    iEnd = sRest.find('_')
    if iEnd < 0:
        iEnd = sRest.find('.')
    if iEnd > 0:
        return sfName[0:iKey] + sRest[iEnd:]
    return sfName[0:iKey]

def AddDesc(sfName, sDesc):
    #print(f'<AddDesc> {sDesc}')
    sAdd = '_' + sDesc
    if sfName.find(sAdd) > 0:
        return sfName
    iDot = sfName.find('.')
    if iDot > 0:
        s = sfName[0:iDot] + sAdd + sfName[iDot:]
    else:
        s = sfName + sAdd
    #print(f'<AdDesc> return {s}')
    return s

def SetType(sfName, sType):
    if sType.find('.') < 0:
        sType = '.' + sType
    if sfName.find(sType) > 0:
        return sfName
    iDot = sfName.find('.')
    if iDot > 0:
        return sfName[0:iDot] + sType
    return sfName + sType

def AddCondExt(sfName, sExt):
    if len(sExt) < 1:
        return sfName
    sDotExt = '.' + sExt
    if sfName.find(sDotExt) > 0:
        return sfName
    return sfName + sDotExt

def VerifyTypeAndExt(sfName, sType, sExt):
    if len(sType) > 0:
        sfName = SetType(sfName, sType)
    return AddCondExt(sfName, sExt)

def CheckGetValue(s, key):
    value = GetValue(s, key)
    print(f'GetValue({s}, {key}) ==> {value}')

def CheckGetValue2(s, key1, key2):
    value = GetValue2(s, key1, key2)
    print(f'GetValue2({s}, {key1}, {key2}) ==> {value}')

def main():
    CheckGetValue('abcd_width55_bla.bla', '_width')
    CheckGetValue2('abcd_mat123_bla.bla', '_mat', '_matrix')
    CheckGetValue2('abcd_matrix1024_bla.bla', '_mat', '_matrix')
    CheckGetValue2('abcd_matrixa1024_bla.bla', '_mat', '_matrix')


if __name__ == '__main__':
    main()
