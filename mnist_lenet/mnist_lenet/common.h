#pragma once
#include <cstdio>



/**
* Computes ceil(x / y) for integral nonnegative values.
*/
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
	return (nominator + denominator - 1) / denominator;
}

/**
* Saves a PGM grayscale image out of unsigned 8-bit data
*/
void SavePGMFile(const unsigned char *data, size_t width, size_t height, const char *filename)
{
	FILE *fp = fopen(filename, "wb");
	if (fp)
	{
		fprintf(fp, "P5\n%lu %lu\n255\n", width, height);
		fwrite(data, sizeof(unsigned char), width * height, fp);
		fclose(fp);
	}
}