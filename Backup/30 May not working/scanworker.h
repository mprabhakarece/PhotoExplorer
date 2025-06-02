#ifndef SCANWORKER_H
#define SCANWORKER_H

#include <QQueue>
#include <QMutex>
#include <QWaitCondition>

struct ScanRequest {
    QString folderPath;
    bool includeSubfolders;
};

extern QQueue<ScanRequest> scanQueue;
extern QMutex scanQueueMutex;
extern QWaitCondition scanQueueNotEmpty;
extern bool scanWorkerRunning;

#endif // SCANWORKER_H
