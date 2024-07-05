package mr

import "log"
import "net"
import "os"
import "net/rpc"
import "net/http"
import "fmt"
import "sync"
import "time"


type Coordinator struct {
	// Your definitions here.
    Filenames []string
    MapTaskQueue []int
    ReduceTaskQueue []int
    ProgressMap map[int]time.Time
    NReduce int
    mu sync.Mutex
    CtMapTaskDone int
    CtReduceTaskDone int
    MapComplete bool
}

// Your code here -- RPC handlers for the worker to call.
func (c *Coordinator) GetInput(args *NullArgs, reply *GetInputReply) error {
    reply.Exit = false
    reply.Wait = false
    reply.NReduce = c.NReduce
    reply.MapTaskCount = len(c.Filenames)

    c.mu.Lock()
    if c.Done() {
        reply.Exit = true
    } else if len(c.MapTaskQueue) > 0 {
        // doing map
        taskIdx := c.MapTaskQueue[0]
        reply.Task = "map"
        reply.Filename = c.Filenames[taskIdx]
        reply.FileNum = taskIdx
        c.ProgressMap[taskIdx] = time.Now()
        c.MapTaskQueue = c.MapTaskQueue[1:]
    } else if c.MapComplete && len(c.ReduceTaskQueue) > 0 {
        //  doing reduce
        reply.Task = "reduce"
        curReduceTaskIdx := c.ReduceTaskQueue[0]
        reply.ReduceTaskIdx = curReduceTaskIdx
        c.ProgressMap[curReduceTaskIdx] = time.Now()
        c.ReduceTaskQueue = c.ReduceTaskQueue[1:]
    } else {
        if !c.MapComplete {
            for taskIdx, startTime := range c.ProgressMap {
                if time.Since(startTime) > 10 * time.Second {
                    fmt.Printf("timeout for map task %v, resetting...\n", taskIdx)
                    c.MapTaskQueue = append(c.MapTaskQueue, taskIdx)
                }
            }
            for _, taskIdx := range c.MapTaskQueue {
                delete(c.ProgressMap, taskIdx)
            }
        } else {
            for taskIdx, startTime := range c.ProgressMap {
                if time.Since(startTime) > 10 * time.Second {
                    fmt.Printf("timeout for reduce task %v, resetting...\n", taskIdx)
                    c.ReduceTaskQueue = append(c.ReduceTaskQueue, taskIdx)
                }
            }
            for _, taskIdx := range c.ReduceTaskQueue {
                delete(c.ProgressMap, taskIdx)
            }
        }
        reply.Wait = true
    }
    c.mu.Unlock()
    return nil
}

func (c *Coordinator) CallTaskDone(args *DoneCallArgs, reply *DoneCallReply) error {
    c.mu.Lock()
    if args.TaskName == "map" {
        c.CtMapTaskDone += 1
        fmt.Printf("map task %v done!\n", args.TaskIdx)
        if c.CtMapTaskDone == len(c.Filenames) {
            c.MapComplete = true
        }
    } else {
        c.CtReduceTaskDone += 1
        fmt.Printf("reduce task %v done!\n", args.TaskIdx)
    }
    delete(c.ProgressMap, args.TaskIdx)
    c.mu.Unlock()

    reply.Ok = true
    return nil
}

//
// an example RPC handler.
//
// the RPC argument and reply types are defined in rpc.go.
//
func (c *Coordinator) Example(args *ExampleArgs, reply *ExampleReply) error {
    fmt.Println("got call!")
    fmt.Println("putting stuff in the reply arg")
	reply.Y = args.X + 1
	return nil
}


//
// start a thread that listens for RPCs from worker.go
//
func (c *Coordinator) server() {
	rpc.Register(c)
	rpc.HandleHTTP()
	//l, e := net.Listen("tcp", ":1234")
	sockname := coordinatorSock()
	os.Remove(sockname)
	l, e := net.Listen("unix", sockname)
	if e != nil {
		log.Fatal("listen error:", e)
	}
	go http.Serve(l, nil)
    fmt.Printf("Server started on %v, listening for incoming RPC\n", sockname)
}

//
// main/mrcoordinator.go calls Done() periodically to find out
// if the entire job has finished.
//
func (c *Coordinator) Done() bool {
	ret := false

	// Your code here.
    if c.CtReduceTaskDone == c.NReduce {
        ret = true
    }

	return ret
}

//
// create a Coordinator.
// main/mrcoordinator.go calls this function.
// nReduce is the number of reduce tasks to use.
//
func MakeCoordinator(files []string, nReduce int) *Coordinator {
	c := Coordinator{}

	// Your code here.
    c.Filenames = files
    c.MapTaskQueue = make([]int, len(c.Filenames))
    for i := range c.MapTaskQueue {
        c.MapTaskQueue[i] = i
    }
    c.NReduce = nReduce
    c.ReduceTaskQueue = make([]int, c.NReduce)
    for i := range c.ReduceTaskQueue {
        c.ReduceTaskQueue[i] = i
    }
    c.ProgressMap = make(map[int]time.Time)
    c.CtMapTaskDone = 0
    c.CtReduceTaskDone = 0
    c.MapComplete = false

	c.server()
	return &c
}
