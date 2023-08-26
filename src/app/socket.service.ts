import { Injectable } from '@angular/core';
import { io, Socket } from 'socket.io-client';
import { Observable } from 'rxjs';
import { HttpClient, HttpParams } from "@angular/common/http";

@Injectable({
  providedIn: 'root'
})
export class SocketService {
  // private socket: Socket;
  // messages = [];
  // onlineUsers = [];
  // nodeURL = 'http://localhost:3000/';
  // constructor(private http: HttpClient) {
  //   this.socket = io('http://localhost:3000');

  //   this.socket.on('chatMessage', (data: string) => {
  //     this.messages.push(data);
  //   });

  //   this.socket.on('userList', (data) => {
  //     console.log(data);
  //     this.onlineUsers.push(data);
  //   });
  // }

  // // Inside SocketService class

  //   getUsers(): Observable<string[]> {
  //     return new Observable<string[]>(observer => {
  //     this.socket.on('userList', users => {
  //       observer.next(users);
  //     });
  //   });
  // }

  // test(){
  //   console.log('test');
  //   return this.http.get<any>(this.nodeURL+'/test');
  // }

  // joinUser(displayName: string) {
  //   this.socket.emit('join', {displayName});
  // }

  // sendPrivateMessage(message: string, displayName: string, recipient: string) {
  //   this.socket.emit('privateMessage', { message, displayName, recipient });
  // }


  // sendMessage(message: string, displayName: string, recipient?: string) {
  //   if(recipient) {
  //     this.sendPrivateMessage(message, displayName, recipient);
  //     return;
  //   }
  //   this.socket.emit('chatMessage', { message, displayName });
  // }

  // getMessage(): Observable<any> {
  //   return new Observable<any>(observer => {
  //     this.socket.on('chatMessage', data => {
  //       observer.next(data);
  //     });
  //   });
  // }
}
