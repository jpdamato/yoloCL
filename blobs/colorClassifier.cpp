
#include "colorClassifier.h"

void RGBtoHSV(float& fR, float& fG, float fB, float& fH, float& fS, float& fV)
{
	float fCMax = MAX(MAX(fR, fG), fB);
	float fCMin = MIN(MIN(fR, fG), fB);
	float fDelta = fCMax - fCMin;

	if (fDelta > 0) {
		if (fCMax == fR) {
			fH = 60 * (fmod(((fG - fB) / fDelta), 6));
		}
		else if (fCMax == fG) {
			fH = 60 * (((fB - fR) / fDelta) + 2);
		}
		else if (fCMax == fB) {
			fH = 60 * (((fR - fG) / fDelta) + 4);
		}

		if (fCMax > 0) {
			fS = fDelta / fCMax;
		}
		else {
			fS = 0;
		}

		fV = fCMax;
	}
	else {
		fH = 0;
		fS = 0;
		fV = fCMax;
	}

	if (fH < 0) {
		fH = 360 + fH;
	}


	fH = fH ;
	fS = fS / 255.0f;
	fV = fV / 255.0f;
}

std::vector<Color> colorPalette;

std::vector<Color> getColorPalette()
{
	if (colorPalette.size() == 0)
	{
		DefinePalletColours();
	}
	return colorPalette;
}

std::vector<Color> DefinePalletColours()
{
colorPalette.clear();
	colorPalette.push_back(Color("black", 255, 0, 0, 0));
	colorPalette.push_back(Color("dark_gray", 255, 64, 64, 64));
	colorPalette.push_back(Color("gray", 255, 128, 128, 128));
	colorPalette.push_back(Color("light_gray", 255, 192, 192, 192));
	colorPalette.push_back(Color("white", 255, 255, 255, 255));
	colorPalette.push_back(Color("red", 255, 200, 0, 0));
	colorPalette.push_back(Color("orange", 255, 240, 105, 0));
	colorPalette.push_back(Color("yellow", 255, 255, 255, 0));
	colorPalette.push_back(Color("lime", 255, 135, 240, 0));
	colorPalette.push_back(Color("green", 255, 0, 185, 40));
	colorPalette.push_back(Color("aquamarine", 255, 0, 240, 105));
	colorPalette.push_back(Color("cyan", 255, 0, 255, 255));
	colorPalette.push_back(Color("light_blue", 255, 0, 135, 230));
	colorPalette.push_back(Color("blue", 255, 0, 0, 180));
	colorPalette.push_back(Color("violet", 255, 95, 0, 180));
	colorPalette.push_back(Color("magenta", 255, 255, 0, 255));
	colorPalette.push_back(Color("fucsia", 255, 200, 0, 75));

	return colorPalette;
}

/// <summary>
///     Colores posibles: black, gray, dark_gray, white, red, orange, white, yellow, green, light_blue, violet, magenta
/// </summary>
/// <param name="c"></param>
/// <returns></returns>
int colorClassify(float r, float g, float b)
{
	// hue 0..360
	//sat 0..1
	//lig 0..1
	float hue, sat, lig;
	RGBtoHSV(r, g, b, hue, sat, lig);


	if (lig <= 0.10 || lig <= 0.15 && sat <= 0.28) return COLOR_PALETTE::black;

	if (lig >= 0.85) return COLOR_PALETTE::white;
	if (lig >= 0.45 && lig <= 0.85 && sat <= 0.22) COLOR_PALETTE::gray;
	if (lig <= 0.25 && sat <= 0.40 ||
		lig <= 0.35 && sat <= 0.15 ||
		lig >= 0.35 && lig <= 0.45 && sat <= 0.05) COLOR_PALETTE::dark_gray;


	if (hue < 26)
		return COLOR_PALETTE::red;
	if (hue < 49)
	{
		if (lig >= 0.80) return COLOR_PALETTE::white;
		return COLOR_PALETTE::orange;
	}
	if (hue < 69)
	{
		if (lig >= 0.80) return COLOR_PALETTE::white;
		return COLOR_PALETTE::yellow;
	}
	if (hue < 153) return COLOR_PALETTE::green;
	if (hue < 190) return COLOR_PALETTE::light_blue;
	if (hue < 260) return COLOR_PALETTE::blue;
	if (hue < 287) return COLOR_PALETTE::violet;
	if (hue < 321) return COLOR_PALETTE::magenta;
	return COLOR_PALETTE::red;

}
int ClosestColor(int r, int g, int b, std::vector<Color>& colorPalette)
{
	int closestColor = 0;
	float minDistance = 100000.0f;
	for (int i = 0; i < colorPalette.size(); i++)
	{
		Color c = colorPalette[i];
		float rDif = r - c.r;
		float gDif = g - c.g;
		float bDif = b - c.b;

		float distance = (float)sqrt(rDif * rDif + gDif * gDif + bDif * bDif);
		if (distance < minDistance)
		{
			closestColor = i;
			minDistance = distance;
		}
	}

	return closestColor;
}
int Closest(Color color, std::vector<Color>& colorPalette)
{
	float minDistance = 100000000;
	int r = color.r;
	int g = color.g;
	int b = color.b;
	return ClosestColor(r, g, b, colorPalette);
}

void CalculatePaletteProportions(cv::Mat& bitmap, float* paletteColorsProportions)
{
	int  totalPixels = 0;

	for (int i = 0; i < PALETTE_COUNT; i++) paletteColorsProportions[i] = 0.0f;


	for (int y = 0; y < bitmap.rows; y++)
		for (int x = 0; x < bitmap.cols; x++)
		{
			uchar c[3];
			c[0] = bitmap.data[y * bitmap.cols * 3 + x * 3];
			c[1] = bitmap.data[y * bitmap.cols * 3 + x * 3 + 1];
			c[2] = bitmap.data[y * bitmap.cols * 3 + x * 3 + 2];
			//string closestColor = Palette.Closest(color);
			int closestColor = colorClassify(c[2], c[1], c[0]); // A partir de HSV
			if (closestColor == COLOR_PALETTE::magenta)
				continue;
			if (closestColor == -1)
				continue;

			paletteColorsProportions[closestColor] = paletteColorsProportions[closestColor] + 1.0f;
			totalPixels++;
		}

	for (int i = 0; i < PALETTE_COUNT; i++) paletteColorsProportions[i] /= totalPixels;


	return;
}