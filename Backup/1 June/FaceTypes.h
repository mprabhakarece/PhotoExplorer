#ifndef FACETYPES_H
#define FACETYPES_H

#include <QString>
#include <QRect>

// Unified structure for face metadata used in DB, UI, and logic
struct FaceEntry {
    int id = -1;
    QString imagePath;      // Full or relative image path
    QRect faceRect;         // Bounding box
    QString globalId;       // e.g., f_00123 (can be empty initially)
    float quality = 0.0f;   // Focus/sharpness score
};

#endif // FACETYPES_H
