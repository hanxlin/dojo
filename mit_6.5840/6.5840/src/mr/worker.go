package mr

import "time"
import "fmt"
import "log"
import "net/rpc"
import "hash/fnv"
import "io/ioutil"
import "os"
import "encoding/json"
import "strconv"
import "sort"

//
// Map functions return a slice of KeyValue.
//
type KeyValue struct {
	Key   string
	Value string
}

// for sorting by key.
type ByKey []KeyValue

// for sorting by key.
func (a ByKey) Len() int           { return len(a) }
func (a ByKey) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByKey) Less(i, j int) bool { return a[i].Key < a[j].Key }

//
// use ihash(key) % NReduce to choose the reduce
// task number for each KeyValue emitted by Map.
//
func ihash(key string) int {
	h := fnv.New32a()
	h.Write([]byte(key))
	return int(h.Sum32() & 0x7fffffff)
}


//
// main/mrworker.go calls this function.
//
func Worker(mapf func(string, string) []KeyValue,
	reducef func(string, []string) string) {

	// Your worker implementation here.

	// uncomment to send the Example RPC to the coordinator.
	// CallExample()
    for {
        getRPCReply := CallGetInput()

        if getRPCReply.Exit == true {
            os.Exit(0)
        }

        if getRPCReply.Wait == true {
            time.Sleep(time.Second)
            continue
        }

        if getRPCReply.Task == "map" {
            var infilename string = getRPCReply.Filename
            var infileNum int = getRPCReply.FileNum
            var nReduce int = getRPCReply.NReduce
            fmt.Printf("Task %v will call map on %v\n", infileNum, infilename)

            // call map over input file
            file, err := os.Open(infilename)
            if err != nil {
                log.Fatalf("cannot open &v", infilename)
            }
            content, err := ioutil.ReadAll(file)
            if err != nil {
                log.Fatalf("cannot read &v", infilename)
            }
            file.Close()
            kva := mapf(infilename, string(content))

            // write to nReduce number of output files
            var ofileArray []*os.File
            var encArray []*json.Encoder
            for i := 0; i < 10; i++ {
                outfilename := fmt.Sprintf("mr-%v-%v", infileNum, strconv.Itoa(i))
                ofile, _ := os.Create(outfilename)
                ofileArray = append(ofileArray, ofile)
                encArray = append(encArray, json.NewEncoder(ofile))
            }
            for _, kv := range kva {
                hashIdx := ihash(kv.Key) % nReduce
                err := encArray[hashIdx].Encode(&kv)
                if err != nil {
                    fmt.Printf("write to file failed")
                }
            }
            for i := 0; i < len(ofileArray); i++ {
                ofileArray[i].Close()
            }

            CallDone("map", infileNum)
        } else if getRPCReply.Task == "reduce"  {
            var reduceTaskIdx, mapTaskCount int = getRPCReply.ReduceTaskIdx, getRPCReply.MapTaskCount

            // read all intermediate files with reduce task idx
            intermediate := []KeyValue{}
            for mapTaskIdx := 0; mapTaskIdx < mapTaskCount; mapTaskIdx++ {
                filename := fmt.Sprintf("mr-%v-%v", strconv.Itoa(mapTaskIdx), strconv.Itoa(reduceTaskIdx))
                file, err := os.Open(filename)
                if err != nil {
                    log.Fatalf("cannot open %v", filename)
                }
                dec := json.NewDecoder(file)
                for {
                    var kv KeyValue
                    if err := dec.Decode(&kv); err != nil {
                        break
                    }
                    intermediate = append(intermediate, kv)
                }
            }

            sort.Sort(ByKey(intermediate))
            fmt.Printf("Intermediate has size %v\n", strconv.Itoa(len(intermediate)))

            oname := "mr-out-" + strconv.Itoa(reduceTaskIdx)
            fmt.Println("Writing output of reduce to", oname)
            ofile, _ := os.Create(oname)

            i := 0
            for i < len(intermediate) {
                j := i + 1
                for j < len(intermediate) && intermediate[j].Key == intermediate[i].Key {
                    j++
                }
                values := []string{}
                for k := i; k < j; k++ {
                    values = append(values, intermediate[k].Value)
                }
                output := reducef(intermediate[i].Key, values)

                // this is the correct format for each line of Reduce output.
                fmt.Fprintf(ofile, "%v %v\n", intermediate[i].Key, output)

                i = j
            }

            CallDone("reduce", reduceTaskIdx)
        } else {
            log.Fatalf("invalid response from coordinator")
        }
    }
}

//
// example function to show how to make an RPC call to the coordinator.
//
// the RPC argument and reply types are defined in rpc.go.
//
func CallExample() {

	// declare an argument structure.
	args := ExampleArgs{}

	// fill in the argument(s).
	args.X = 99

	// declare a reply structure.
	reply := ExampleReply{}

	// send the RPC request, wait for the reply.
	// the "Coordinator.Example" tells the
	// receiving server that we'd like to call
	// the Example() method of struct Coordinator.
	ok := call("Coordinator.Example", &args, &reply)
	if ok {
		// reply.Y should be 100.
		fmt.Printf("reply.Y %v\n", reply.Y)
	} else {
		fmt.Printf("call failed!\n")
	}
}

func CallDone(taskName string, taskIdx int) bool {
    args := DoneCallArgs{}
    args.TaskName = taskName
    args.TaskIdx = taskIdx
    reply := DoneCallReply{}

	ok := call("Coordinator.CallTaskDone", &args, &reply)
	if !ok {
		fmt.Printf("call failed!\n")
	}

    return reply.Ok
}

func CallGetInput() *GetInputReply {
	args := NullArgs{}
    reply := GetInputReply{}

	ok := call("Coordinator.GetInput", &args, &reply)
	// if ok {
	// 	fmt.Printf("reply.IsMapTask %v\n", reply.IsMapTask)
	// } else {
	// 	fmt.Printf("call failed!\n")
	// }
	if !ok {
		fmt.Printf("call failed!\n")
	}

    return &reply
}

//
// send an RPC request to the coordinator, wait for the response.
// usually returns true.
// returns false if something goes wrong.
//
func call(rpcname string, args interface{}, reply interface{}) bool {
	// c, err := rpc.DialHTTP("tcp", "127.0.0.1"+":1234")
	sockname := coordinatorSock()
	c, err := rpc.DialHTTP("unix", sockname)
	if err != nil {
		log.Fatal("dialing:", err)
	}
	defer c.Close()

	err = c.Call(rpcname, args, reply)
	if err == nil {
		return true
	}

	fmt.Println(err)
	return false
}
