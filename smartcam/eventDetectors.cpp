#include "eventDetectors.h"
#include "../blobs/Export.h"
/*
#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include "scImage.h"
#include "trackingLib.h"


using namespace trackingLib;

static EventManager* _emInstance;
static int _eventIDCreator = 0;
int maxElapsedTime = 20000;
//Minimo de detecciones para considerarse como un blob valido
int minDetections = 5;
Event::Event() { id = -1; type = -1;  trackedObjectId = -1; idDetector = ""; }
Event::Event(int _id, int _type, string trackedOId, string _mess, Blob* b, int nroframe, cv::Rect boundingbox, string idDetectorAlgorithm, cv::Point pos, std::chrono::time_point<std::chrono::system_clock> now, Mat _frame)
{
	id = _id; 
	type = _type;
	//boost::uuids::uuid uuid = boost::uuids::random_generator()();
	eventUID = "00AAS000";// to_string(uuid);
	trackedObjectId = trackedOId;
	blob = b;
	nroFrame = nroframe;
	boundingBox = boundingbox;
	idDetector = idDetectorAlgorithm;
	message = _mess;
	datetime = now;
	position = pos;
	frame = _frame;
}

string Event::addQuotes(std::string s)
{
	return  "\"" + s + "\"";
}

std::string Event::getAsJSON()
{
	std::string _json = "";
	auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(datetime);
	auto epoch = now_ms.time_since_epoch();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
	//milisegundos desde DateTime(1970, 1, 1)
	_json = _json + " { " + "\n" +
		"\t\t" + addQuotes("event_id") + ": " + addQuotes(eventUID) + "," + "\n" +
		"\t\t" + addQuotes("id_detector") + ": " + addQuotes(idDetector) + "," + "\n" +
		"\t\t" + addQuotes("id_trackedblob") + ": " + std::to_string(blob->id) + "," + "\n" +
		"\t\t" + addQuotes("guid_trackedblob") + ": " + addQuotes(blob->guid) + "," + "\n" +
		"\t\t" + addQuotes("type") + ": " + std::to_string(type) + "," + "\n" +
		"\t\t" + addQuotes("nroframe") + ": " + std::to_string(nroFrame) + "," + "\n" +
		"\t\t" + addQuotes("miliseconds") + ": " + std::to_string(ms.count()) + "," + "\n" +
		"\t\t" + addQuotes("position_x") + ": " + std::to_string(relativePositionX(position.x)) + "," + "\n" +
		"\t\t" + addQuotes("position_y") + ": " + std::to_string(relativePositionY(position.y)) + "," + "\n" +
		"\t\t" + addQuotes("width") + ": " + std::to_string(relativePositionX(boundingBox.width)) + "," + "\n" +
		"\t\t" + addQuotes("height") + ": " + std::to_string(relativePositionY(boundingBox.height)) + "," + "\n" +
		"\t\t" + addQuotes("message") + ": " + addQuotes(message) +  "\n"
		"}";
	
	return _json;
}

void Event::save(std::string dir, std::string name)
{
	std::string json = getAsJSON();
	// save the xml document
	std::ofstream xml_file;

	if (endsWith(dir, "\\") == false)
		dir.append("\\");
	xml_file.open(dir + name+".json");
	xml_file << json;
	xml_file.close();
}

bool Event::similar(Event e)
{
	if ((e.type == type) && (e.blob == blob))
		return true;
	else
		return false;

}
// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool _intersection(cv::Point o1, cv::Point p1, cv::Point o2, cv::Point p2)
{
	float ax = (float)p1.x - o1.x;     // direction of line a
	float ay = (float)p1.y - o1.y;     // ax and ay as above

	float bx = (float)o2.x - p2.x;     // direction of line b, reversed
	float by = (float)o2.y - p2.y;     // really -by and -by as above

	float dx = (float)o2.x - o1.x;   // right-hand side
	float dy = (float)o2.y - o1.y;

	float det = (float)(ax * by - ay * bx);

	if (det == 0) return false;

	float r = (dx * by - dy * bx) / det;
	float s = (ax * dy - ay * dx) / det;

	return !(r < 0 || r > 1 || s < 0 || s > 1);
}

int Event::getID()
{
	_eventIDCreator++;
	return _eventIDCreator;

}


/*Dada una lista de activeblobs determina si alguno cruzó la linea, y lo retorna en detectedEvents*/
std::vector<Event> LineCrossDetector::detect(vector<Blob*>& activeblobs, Mat& frame, int nFrame, std::chrono::time_point<std::chrono::system_clock> now)
{
	//detectedEvents.clear();
	cv::Point istart((int)(start.x * frame.cols), (int)(start.y*frame.rows));
	cv::Point iend((int)(end.x * frame.cols), (int)(end.y*frame.rows));

	for (int i = 0; i < activeblobs.size(); i++)
	{
		Blob* b = activeblobs[i];
		if ((b->life > 0) && (b->before.size() > MIN_OBJECT_HISTORY))
		{
			Blob* b0 = b->before[b->before.size() - 2];
			Blob* b1 = b->before[b->before.size() - 1];
			cv::Point c0 = b0->getCenter();
			cv::Point c1 = b1->getCenter();

			if (_intersection(istart, iend, c0, c1))
			{
				wasCrossed = 10;				
			    cout << nFrame << " || Cross line event. Object " << b->id << " Center " << c0.x << " " << c0.y << " " << c1.x << " "<< c1.y << "\n";
				Event e(1, EVENT_TYPE_CROSS_LINE, b->guid, "Cross Line ", b, nFrame, b0->getRect(), this->id, b0->getCenter(),now,frame);
			
				cv::Mat cut = frame(b0->getRect()).clone();
				cv::resize(cut, cut, cv::Size(), 2.0, 2.0);
				e.frame = cut;

				double d0 =	(iend-istart).cross(c0-istart);
				double d1 = (iend - istart).cross(c1 - istart);

				if (d0 > 0 && d1 < 0)
				{
					e.message += "A->B";
				}
				else
				{
					e.message += "B->A";
				}
				detectedEvents.push_back(e);
			}
		}
	}

	return detectedEvents;
}

void LineCrossDetector::drawOn(Mat& frame)
{
	cv::Point istart((int)(start.x * frame.cols), (int)(start.y*frame.rows));
	cv::Point iend((int)(end.x * frame.cols), (int)(end.y*frame.rows));

	cv::line(frame, istart, iend, cv::Scalar(0, 0, 255), 2);
	std::string mess = to_string(this->detectedEvents.size());
	cv::putText(frame, mess, istart, 1, 2, cv::Scalar(50, 50, 200), 2);

	
}


void SceneDetector::CheckTimeOnScene(int nFrame, std::chrono::time_point<std::chrono::system_clock> now, int i, Blob* b, Mat frame)
{
	if (!b->longTime) //Si el objeto aun no fue notificado como que estuvo mucho tiempo en escena
	{
		auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now - b->creationTimePoint);

	//	auto  elapsedTime = now - b->creationTimePoint;
		if (milliseconds.count() > maxElapsedTime && (b->before.size() > minDetections))
		{
			//std::cout << nFrame << " || Long time in zone  event. Object " << innerblobs[i]->id << " Center " << b->getCenter().x << " " << b->getCenter().y << "\n";
			b->longTime = true;
			Event e(1, EVENT_TYPE_LONGTIME_SCENE, b->guid, "Long time", b, nFrame, b->getRect(), this->id, b->getCenter(), now, frame);
			detectedEvents.push_back(e);
		}
	}	
}

std::vector<Event> AreaDetector::detect(vector<Blob*>& activeblobs, Mat& frame, int nFrame, std::chrono::time_point<std::chrono::system_clock> now)
{
	detectedEvents.clear();

	std::vector<cv::Point2f> icontour;
	for (int i = 0; i < contour.size(); i++)
	{
		icontour.push_back(cv::Point( (int)(contour[i].x * frame.cols), (int)(contour[i].y*frame.rows)));
	}
	
	for (int i = 0; i < activeblobs.size(); i++)
	{
		Blob* b = activeblobs[i];
		if (b->life <= 0) //el objeto ya no se traquea, reviso si se perdió dentro del area
		{
			Blob* bLast = b->before[b->before.size() - 1];
			//Positivo es adentro, negativo afuera, cero contorno
			double dst0 = pointPolygonTest(icontour, bLast->getCenter(), true);
			if (dst0 >= 0)
			{
				EventManager::getInstance()->addNewEvent(Event(0, EVENT_TYPE_DISSAPEAR_AREA, b->guid, "Object dissapear", b, nFrame, bLast->getRect(), this->id, b->getCenter(), now, frame));
			}
		}
		else 
		{
			if (b->before.size() >= minDetections) //Si tiene un historial de detecciones que supere el mínimo
			{
				//CheckTimeOnScene(nFrame, now, i, b);

				// Chequeo los 2 ultimos estados
				Blob* bPrevLast = b->before[b->before.size() - 2];
				Blob* bLast = b->before[b->before.size() - 1];				
				//Positivo es adentro, negativo afuera, cero contorno
				double dstPrevLast = pointPolygonTest(icontour, bPrevLast->getCenter(), true);
				double dstLast = pointPolygonTest(icontour, bLast->getCenter(), true);
				// Antes estaba adentro y ahora afuera o en contorno
				if ((dstPrevLast > 0) && (dstLast <= 0))
				{
					// blob not in vector
					b->timeExitZone = now;
					// Lo agrego como nuevo objeto del area
					innerblobs.push_back(b);  //QUITARLO?????
					//cout << nFrame << " || exiting zone  event. Object " << b->id << " Center " << b->getCenter().x << " " << b->getCenter().y << "\n";
					Event e(0, EVENT_TYPE_LEAVE_AREA, b->guid, "Exiting zone", b, nFrame, bLast->getRect(), this->id, bLast->getCenter(), now,frame);
					detectedEvents.push_back(e);
					wasCrossed = 10;
				}
				else
				{
					// Antes estaba afuera y ahora adentro
					if ((dstPrevLast <= 0) && (dstLast > 0))
					{
						// blob not in vector
						b->timeEnterZone = now;
						// Lo agrego como nuevo objeto del area
						innerblobs.push_back(b);
						//cout << nFrame << " || entering zone  event. Object " << b->id << " Center " << b->getCenter().x << " " << b->getCenter().y << "\n";
						Event e(0, EVENT_TYPE_ENTER_AREA, b->guid, "Entering zone", b, nFrame, bLast->getRect(), this->id, bLast->getCenter(), now,frame);
						detectedEvents.push_back(e);
						wasCrossed = 10;
					}					
				}

			}
		}
		

	}


	// Limpio objetos anteriores
	for (int i = 0; i < innerblobs.size(); i++)
	{
		Blob* b = innerblobs[i]->getLast();
		double dst = pointPolygonTest(icontour, b->getCenter(), true);		
		if (b->life < 0) // El objeto lo deje de trackear
		{
			//std::cout << nFrame << " || Dissapear . Object " << innerblobs[i]->id << " Center " << b->getCenter().x << " " << b->getCenter().y << "\n";
			// Lo remuevo de la lista			
			Event e(1, EVENT_TYPE_LEAVE_SCENE, innerblobs[i]->guid, "Object dissapear", innerblobs[i],nFrame,b->getRect(), this->id,b->getCenter(),now,frame);
			detectedEvents.push_back(e);
			innerblobs[i] = NULL;
		}
		//Esto va si solo se considera que salen aquellos que antes ingresaron
		//else  
		//// Si el objeto salio, mando el evento
		//if (dst < 0)
		//{
		//	std::cout << nFrame << " || Leaving zone  event. Object " << innerblobs[i]->id << " Center " << b->getCenter().x << " " << b->getCenter().y << "\n";
		//	// Lo remuevo de la lista
		//	

		//	Event e(1, 1, innerblobs[i]->id, "Object leaved", innerblobs[i], nFrame, b->r, this, b->getCenter(), now);		
		//	detectedEvents.push_back(e);

		//	innerblobs[i] = NULL;
		//}		
	}
	// Elimino los objetos nulos
	vector<Blob*> temp;
	innerblobs.swap(temp);
	innerblobs.clear();

	for (int j = 0; j < temp.size(); j++)
	{
		if (temp[j] != NULL)
			innerblobs.push_back(temp[j]);

	}

	return detectedEvents;
}

void AreaDetector::drawOn(Mat& frame)
{
	std::vector<cv::Point> icontour;
	for (int i = 0; i < contour.size(); i++)
	{
		icontour.push_back(cv::Point((int)(contour[i].x * frame.cols) , (int)(contour[i].y*frame.rows)));
	}

	if (wasCrossed > 0)
	{
		cv::polylines(frame, icontour, true, cv::Scalar(255, 0, 0), 2);
		wasCrossed--;
	}
	else
	{
		cv::polylines(frame, icontour, true, cv::Scalar(0, 0, 255), 2);
	}
}

std::vector<Event> SceneDetector::detect(vector<Blob*>& activeblobs, Mat& frame, int nFrame, std::chrono::time_point<std::chrono::system_clock> now)
{

	detectedEvents.clear();

	for (int i = 0; i < activeblobs.size(); i++)
	{
		Blob* b = activeblobs[i];
		if (b->life > 0)
		{
			if (b->before.size() >= minDetections) //Si tiene un historial de detecciones que supere el mínimo
			{
				CheckTimeOnScene(nFrame, now, i, b,frame);
			}
		}
	}

	return detectedEvents;
}

void SceneDetector::drawOn(Mat& frame)
{
	//std::vector<cv::Point> icontour;
	//for (int i = 0; i < contour.size(); i++)
	//{
	//	icontour.push_back(cv::Point((int)(contour[i].x * frame.cols), (int)(contour[i].y*frame.rows)));
	//}

	//if (wasCrossed > 0)
	//{
	//	cv::polylines(frame, icontour, true, cv::Scalar(255, 0, 0), 2);
	//	wasCrossed--;
	//}
	//else
	//{
	//	cv::polylines(frame, icontour, true, cv::Scalar(0, 0, 255), 2);
	//}
}




EventManager* EventManager::getInstance()
{
	if (_emInstance == NULL)
		_emInstance = new EventManager();
	return _emInstance;
}
void EventManager::addNewEvent(Event e)
{
	e.id = Event::getID();
	// Si aun tengo un evento similar, lo descarto. Puede ser que dos barreras disparen dos eventos de un mismo objeto, por ej si tenemos una barrera dentro de un area, o dos barreras juntas. 
	for (int i = 0; i < 10; i++)
	{
		if (lastEvents[i].similar(e))
		{
			std::cout << "Se descartó el evento porque ya hay uno similar";
			return;
		}			
	}
	events.push_back(e);
	lastEvents[innerIndex] = e;
	innerIndex = (innerIndex + 1) % 10;

}

void EventManager::addNewEvents(std::vector<Event> es)
{
	for (int i = 0; i < es.size(); i++)
	{
		addNewEvent(es[i]);
	}
}

void EventManager::printError(string s)
{
	std::cout << s;
}