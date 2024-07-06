package kvsrv

import (
	"log"
	"sync"
)

const Debug = false

func DPrintf(format string, a ...interface{}) (n int, err error) {
	if Debug {
		log.Printf(format, a...)
	}
	return
}


type ReqCache struct {
    reqId int
    prevGetCache string
    prevAppendCache string
}

type KVServer struct {
	mu sync.Mutex

	// Your definitions here.
    data map[string]string
    clientReq map[int64]*ReqCache
}


func (kv *KVServer) Get(args *GetArgs, reply *GetReply) {
	// Your code here.
    kv.mu.Lock()
    defer kv.mu.Unlock()

    val := kv.data[args.Key]
    reply.Value = val
}

func (kv *KVServer) Put(args *PutAppendArgs, reply *PutAppendReply) {
	// Your code here.
    kv.mu.Lock()
    defer kv.mu.Unlock()
    clientId := args.ClientId
    reqId := args.ReqId
    prevCache, ok := kv.clientReq[clientId]
    if !ok || prevCache.reqId != reqId {
        // this is a new request, clear out old cache
        kv.clientReq[clientId] = &ReqCache{reqId: reqId}
        kv.data[args.Key] = args.Value
    }
}

func (kv *KVServer) Append(args *PutAppendArgs, reply *PutAppendReply) {
	// Your code here.
    kv.mu.Lock()
    defer kv.mu.Unlock()
    clientId := args.ClientId
    reqId := args.ReqId
    prevCache, ok := kv.clientReq[clientId]
    if !ok || prevCache.reqId != reqId {
        // this is a new request, clear out old cache
        kv.clientReq[clientId] = &ReqCache{reqId: reqId}
        var oldVal string
        oldVal, ok := kv.data[args.Key]
        if !ok {
            kv.data[args.Key] = args.Value
        } else {
            kv.data[args.Key] = oldVal + args.Value
        }
        kv.clientReq[clientId].prevAppendCache = oldVal
        reply.Value = oldVal
    } else {
        reply.Value = kv.clientReq[clientId].prevAppendCache
    }
}

func StartKVServer() *KVServer {
	kv := new(KVServer)

	// You may need initialization code here.
    kv.data = make(map[string]string)
    kv.clientReq = make(map[int64]*ReqCache)

	return kv
}
