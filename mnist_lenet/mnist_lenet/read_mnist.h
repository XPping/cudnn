#pragma once

#include<cstdint>
#include<cstddef>


/*
* image_filename: image ubyte file name path
* label_filename: label ubyte file name path
* images: readed images data
* labels: readed labels data
* width: image width
* height: image height
return: the number of image
*/
size_t readUbyteMnist(const char* image_filename, const char* label_filename,
	uint8_t *images, uint8_t *labels, size_t& width, size_t& height);