#pragma once

#ifndef COLOR_CLASSIFIER_H
#define COLOR_CLASSIFIER_H

#include <cstdio>
#include <vector>
#include <stdio.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define PALETTE_COUNT 17

typedef enum {
	black, dark_gray,gray, light_gray, white, red, orange,  yellow, lime, green,aquamarine, cyan, light_blue, blue,  violet, magenta, fucsia
} COLOR_PALETTE;

static std::string COLOR_PALETTE_NAMES[] = {
	"black", "dark_gray", "gray", "light_gray", "white", "red", "orange", "yellow", "lime", "green", "aquamarine", "cyan", "light_blue", "blue", "violet", "magenta", "fucsia" } ;

struct Color
{
public:
	float r, g, b;
	float h, s, v;
	std::string name;
	float getHue() { 		return h; 	};
	float getSaturation() { return s; };
	float getBrightness() { return v; };
	Color(char* _name, float a, float _r, float _g, float _b) { name = _name; r = _r; g = _g; b = _b; }
	Color(float _r, float _g, float _b) { r = _r; g = _g; b = _b; }
};

std::vector<Color> getColorPalette();
void RGBtoHSV(float& fR, float& fG, float fB, float& fH, float& fS, float& fV);
std::vector<Color> DefinePalletColours();
/// <summary>
///     Colores posibles: black, gray, dark_gray, white, red, orange, white, yellow, green, light_blue, violet, magenta
/// </summary>
/// <param name="c"></param>
/// <returns></returns>
int colorClassify(float r, float g, float b);


int ClosestColor(int r, int g, int b, std::vector<Color>& colorPalette);
int Closest(Color color, std::vector<Color>& colorPalette);

void CalculatePaletteProportions(cv::Mat& bitmap, float* paletteColorsProportions);

#endif
