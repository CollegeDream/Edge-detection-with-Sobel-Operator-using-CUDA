#pragma once
#ifndef _RGB_H_
#define _RGB_H_

typedef int LONG;
typedef unsigned short WORD;
typedef unsigned int DWORD;
typedef unsigned char BYTE;

#define min(X,Y)	(X < Y ? X : Y);
#define max(X,Y)	(X >= Y ? X : Y);
/**
*  Structure used to store the Red, Green and Blue primary color intensities in a pixel
*  Each color component is of type BYTE(unsigned char), thus the maximum value for each component is 0xFF (255)
*/
struct RGB { // A struct is like a class, but with all its members public (remember data structures!)
	BYTE red;
	BYTE green;
	BYTE blue;
};

#endif