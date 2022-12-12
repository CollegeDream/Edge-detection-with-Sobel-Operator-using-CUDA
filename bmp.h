#pragma once
#ifndef _BMP_H_
#define _BMP_H_

#pragma pack(2)


#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "RGB.h"

/**
*  A BMP file always begins with a BITMAPFILEHEADER
*  This header store general information about the BMP file
*/
typedef struct tagBITMAPFILEHEADER {
	WORD  bfType;		// must be 'BM' 
	DWORD bfSize;		// size of the whole .bmp file
	WORD  bfReserved1;	// must be 0
	WORD  bfReserved2;	// must be 0
	DWORD bfOffBits;	// offset, in bytes, from the BITMAPFILEHEADER structure to the bitmap bits
} BITMAPFILEHEADER, * PBITMAPFILEHEADER;

/**
* A BMP file will have a BITMAPINFOHEADER specified after the BITMAPFILEHEADER
* This header stores specific information about the BMP image
*/
typedef struct tagBITMAPINFOHEADER {
	DWORD biSize;			// size of the structure
	LONG  biWidth;			// image width
	LONG  biHeight;			// image Height
	WORD  biPlanes;			// bitplanes
	WORD  biBitCount;		// resolution
	DWORD biCompression;	// compression
	DWORD biSizeImage;		// size of the image
	LONG  biXPelsPerMeter;	// pixels per meter X
	LONG  biYPelsPerMeter;	// pixels per meter Y
	DWORD biClrUsed;		// colors used
	DWORD biClrImportant;	// important colors
} BITMAPINFOHEADER, * PBITMAPINFOHEADER;

/**
* Bitmap class: supports functions for reading and writing to bmp files
* @author: Ed Walker
*/
class BitMap {
private:

	BYTE* bmp_buffer; // byte array holding the contents of the entire bitmap file
	DWORD bmp_size;   // size of the bmp_buffer array

	PBITMAPFILEHEADER file_header;
	PBITMAPINFOHEADER info_header;

	/**
	* Converts a byte array in BMP format to RGB format
	*/
	BYTE* convertBMPtoRGB(BYTE* buffer, int width, int height) {
		int padding = 0;
		int scanlinebytes = width * 3;
		while ((scanlinebytes + padding) % 4 != 0)
			++padding;

		int psw = scanlinebytes + padding;

		BYTE* newbuf = new BYTE[width * height * 3];

		size_t bufpos = 0;
		size_t newpos = 0;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < 3 * width; x += 3) {
				newpos = y * 3 * width + x;
				bufpos = (height - y - 1) * psw + x;

				newbuf[newpos] = buffer[bufpos + 2];
				newbuf[newpos + 1] = buffer[bufpos + 1];
				newbuf[newpos + 2] = buffer[bufpos];
			}
		}

		return newbuf;
	}


	/**
	* Converts a byte array from RGB format to a BMP format
	* Note that the BMP format pads each row with 0s in order to align it by 4 bytes, therefore
	* a BMP array may be bigger than a RGB array because of this padding.  Consequently, we need to
	* return the new size in the newsize parameter.
	*/
	BYTE* convertRGBtoBMP(BYTE* buffer, int width, int height, size_t* newsize) {
		int padding = 0;
		int scanlinebytes = width * 3;
		while ((scanlinebytes + padding) % 4 != 0)
			padding++;

		int psw = scanlinebytes + padding;

		*newsize = height * psw;
		BYTE* newbuf = new BYTE[*newsize];

		memset(newbuf, 0, *newsize);

		size_t bufpos = 0;
		size_t newpos = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < 3 * width; x += 3) {
				bufpos = y * 3 * width + x;     // position in original buffer
				newpos = (height - y - 1) * psw + x; // position in padded buffer
				newbuf[newpos] = buffer[bufpos + 2];       // swap r and b
				newbuf[newpos + 1] = buffer[bufpos + 1]; // g stays
				newbuf[newpos + 2] = buffer[bufpos];     // swap b and r
			}
		}

		return newbuf;
	}

public:

	/**
	* Constructor for initializing the object with the contents of a BMP array
	*/
	BitMap(const std::string& filename) {
		std::ifstream file(filename, ios::binary);

		if (file.fail()) {
			std::cerr << "Fail to open " << filename << " for reading\n";
			return;
		}

		file.seekg(0, std::ios::end); // move the file pointer to the end of the file
		std::streamoff length = file.tellg(); // get the offset at the end of the file --> length of the file
		file.seekg(0, std::ios::beg); // move the file pointer back to the front of the file

		bmp_buffer = new BYTE[length]; // create an array for storing the contents of the file
		bmp_size = length;
		file.read((char*)bmp_buffer, length); // read the entire file

		file_header = (PBITMAPFILEHEADER)(&bmp_buffer[0]); // start of the file contains the BITMAPFILEHEADER
		info_header = (PBITMAPINFOHEADER)(&bmp_buffer[0] + sizeof(BITMAPFILEHEADER)); // closely followed by the BITMAPINFOHEADER
	}

	/**
	* Destructor
	*/
	~BitMap() {
		delete[] bmp_buffer; // done with array, so return it
	}

	/**
	* Returns the height of the image from BITMAPINFOHEADER
	*/
	int getHeight() {
		return info_header->biHeight;
	}

	/**
	* Returns the width of the image from BITMAPINFOHEADER
	*/
	int getWidth() {
		return info_header->biWidth;
	}

	/**
	* Returns the image size from BITMAPINFOHEADER
	*/
	int getImageSize() {
		return info_header->biSizeImage;
	}

	/**
	* Return the bit encoding length from BITMAPINFOHEADER
	*/
	int getBitCount() {
		return info_header->biBitCount;
	}

	/**
	* Returns the pixel array in RGB format
	*/
	RGB* getRGBImageArray() {
		int size = file_header->bfSize - sizeof(BITMAPFILEHEADER) - sizeof(BITMAPINFOHEADER);

		return (RGB*)convertBMPtoRGB(&bmp_buffer[file_header->bfOffBits], info_header->biWidth, info_header->biHeight);
	}

	/**
	* Saves the RGB array in the BMP image array
	*/
	void setImageFromRGB(RGB* buffer) {
		size_t size = 0;
		BYTE* newbuffer = convertRGBtoBMP((BYTE*)buffer, info_header->biWidth, info_header->biHeight, &size);

		memcpy(&bmp_buffer[file_header->bfOffBits], newbuffer, size);
	}

	/**
	* Saves this BitMap object in a file
	*/
	void saveBMP(const char* filename) {
		std::ofstream fout(filename, ios::binary);
		if (fout.fail()) {
			std::cerr << "Fail to open file " << filename << " for writing\n";
			return;
		}

		fout.write((char*)file_header, sizeof(BITMAPFILEHEADER));
		fout.write((char*)info_header, sizeof(BITMAPINFOHEADER));

		fout.write((char*)&bmp_buffer[file_header->bfOffBits], bmp_size - file_header->bfOffBits);
		fout.close();
		return;
	}
};

#endif